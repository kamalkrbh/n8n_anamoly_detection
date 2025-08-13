"""Simple anomaly detection and LLM summarization pipeline for router metrics in InfluxDB.

This module:
1. Queries recent time-series metrics (if_octets_in/out, cpu_utilization, packet_loss, latency_ms, connections_active)
2. Performs basic statistical & contextual anomaly detection:
   - Z-score spikes
   - Percentage thresholds (e.g., packet loss > 2%)
   - Time-of-day contextual anomalies (high traffic at night)
3. Produces a structured anomaly report
4. (Optional) Formats a prompt for an external LLM (Groq/Ollama/OpenAI) without making the call (user plugs in provider)

Usage (example):
    python anomaly_detector.py --start "2025-08-01 00:00:00" --end "2025-08-08 00:00:00" --bucket network_data --org your-org \
        --url http://localhost:8086 --token $INFLUXDB_TOKEN --llm-prompt out_prompt.txt

Extend: Replace build_llm_prompt() or integrate an actual client.
"""
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException

# Load variables from .env (if present) before reading with os.getenv in arg defaults
load_dotenv()

DEFAULT_METRICS = [
    "if_octets_in",
    "if_octets_out",
    "cpu_utilization",
    "packet_loss",
    "latency_ms",
    "connections_active",
]

NIGHT_HOURS = set(list(range(0,6)) + [23])  # 11pm-5:59am considered "night" context

@dataclass
class Anomaly:
    metric: str
    device: str
    time: datetime
    severity: str
    reason: str
    value: float
    baseline_mean: float
    baseline_std: float
    zscore: float
    context: Dict[str, Any]

@dataclass
class AnomalyReport:
    start: datetime
    end: datetime
    generated_at: datetime
    anomalies: List[Anomaly]

# ---------------- Query Layer -----------------

def query_metric_frames(client: InfluxDBClient, bucket: str, org: str, start: datetime, end: datetime, metrics: List[str]) -> pd.DataFrame:
    """Query all metrics separately then concat to a tall DataFrame: columns: [time, metric, device, value]."""
    frames = []
    for metric in metrics:
        flux = f"""
from(bucket: "{bucket}")
  |> range(start: {start.isoformat()}, stop: {end.isoformat()})
  |> filter(fn: (r) => r._measurement == \"{metric}\")
  |> filter(fn: (r) => r._field == \"value\")
  |> keep(columns: [\"_time\",\"_value\",\"device\"])
        """
        tables = client.query_api().query(flux, org=org)
        rows: list[dict] = []
        for table in tables:
            for record in table.records:
                try:
                    rec_time = record.get_time()
                except KeyError:
                    rec_time = record.values.get("_time") or record.values.get("time")
                try:
                    rec_value = record.get_value()
                except KeyError:
                    rec_value = record.values.get("_value") or record.values.get("value")
                rows.append({
                    "time": rec_time,
                    "value": rec_value,
                    "device": record.values.get("device"),
                    "metric": metric
                })
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["time","metric","device","value"])  # empty
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("time", inplace=True)
    return df

# --------------- Detection --------------------

def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per metric+device global mean and std for reference."""
    stats = df.groupby(["metric","device"]).agg(mean=("value","mean"), std=("value","std")).reset_index()
    return stats

def detect_anomalies(df: pd.DataFrame, stats: pd.DataFrame) -> List[Anomaly]:
    anomalies: List[Anomaly] = []
    if df.empty:
        return anomalies
    merged = df.merge(stats, on=["metric","device"], how="left")
    # Avoid division by zero
    merged["std"].replace({0: np.nan}, inplace=True)
    merged["zscore"] = (merged["value"] - merged["mean"]) / merged["std"]

    severity_rank = {"minor": 1, "major": 2, "critical": 3}

    def escalate(current: Optional[str], new: str) -> str:
        if current is None:
            return new
        return new if severity_rank[new] > severity_rank[current] else current

    for _, row in merged.iterrows():
        sev: Optional[str] = None
        reasons: List[str] = []
        metric = str(row.get("metric"))
        raw_val = row.get("value")
        if raw_val is None:
            continue
        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            continue
        mean = row.get("mean")
        std = row.get("std")
        if pd.isna(mean):
            mean = np.nan
        if pd.isna(std):
            std = np.nan
        z = row.get("zscore")
        if pd.isna(z):
            z = np.nan
        # Generic z-score thresholds
        if not np.isnan(z):
            if z >= 4:
                sev = escalate(sev, "critical"); reasons.append(f"z>=4 spike vs mean {mean:.2f}")
            elif z >= 3:
                sev = escalate(sev, "major"); reasons.append("z>=3 spike")
            elif z <= -3:
                sev = escalate(sev, "major"); reasons.append("z<=-3 drop")
        # Metric-specific rules
        if metric == "cpu_utilization" and value >= 90:
            sev = escalate(sev, "critical"); reasons.append("CPU >=90%")
        if metric == "packet_loss" and value >= 2:
            sev = escalate(sev, "critical"); reasons.append("loss >=2%")
        if metric == "latency_ms" and value >= 150:
            sev = escalate(sev, "major"); reasons.append("latency >=150ms")
        if metric.startswith("if_octets_") and value >= 50_000_000:
            sev = escalate(sev, "major"); reasons.append("bandwidth very high")
        if metric == "connections_active" and value >= 5000:
            sev = escalate(sev, "major"); reasons.append("connections surge")

        t = row.get("time")
        if isinstance(t, pd.Timestamp):
            ts = t.to_pydatetime()
        else:
            ts = t if isinstance(t, datetime) else None
        hour = ts.hour if ts else None
        if hour is not None and hour in NIGHT_HOURS and not np.isnan(std):
            if metric in ("if_octets_in", "if_octets_out") and value > mean + 2 * std:
                sev = escalate(sev, "minor"); reasons.append("night traffic spike")
            if metric == "cpu_utilization" and value > mean + 2 * std:
                sev = escalate(sev, "minor"); reasons.append("night CPU spike")
        if sev and ts:
            anomalies.append(Anomaly(
                metric=metric,
                device=str(row.get("device")),
                time=ts,
                severity=sev,
                reason="; ".join(reasons),
                value=value,
                baseline_mean=0.0 if np.isnan(mean) else float(mean),
                baseline_std=0.0 if np.isnan(std) else float(std),
                zscore=0.0 if np.isnan(z) else float(z),
                context={"hour": hour}
            ))
    return anomalies

# --------------- Reporting / LLM Prompt ---------------

def summarize_anomalies(anomalies: List[Anomaly]) -> Dict[str, Any]:
    sev_counts: Dict[str,int] = {}
    for a in anomalies:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1
    by_metric: Dict[str,int] = {}
    for a in anomalies:
        by_metric[a.metric] = by_metric.get(a.metric, 0) + 1
    return {"severity_counts": sev_counts, "metric_counts": by_metric, "total": len(anomalies)}

def build_llm_prompt(report: AnomalyReport) -> str:
    summary = summarize_anomalies(report.anomalies)
    lines = [
        "You are an expert network operations analyst. Analyze the detected anomalies.",
        f"Time window: {report.start.isoformat()} to {report.end.isoformat()} UTC",
        f"Total anomalies: {summary['total']}; Severity breakdown: {summary['severity_counts']}",
        "List of anomalies (metric device time severity value zscore reason):",
    ]
    for a in report.anomalies[:200]:  # cap to avoid huge prompts
        lines.append(f"- {a.metric} {a.device} {a.time.isoformat()} {a.severity} value={a.value:.2f} z={a.zscore:.2f} reason={a.reason}")
    lines.append("\nTasks: 1) Classify overall health. 2) Highlight most critical root causes. 3) Suggest likely causes. 4) Recommend remediation & prioritization. 5) Note any time-of-day anomalies (e.g., unusual night traffic).")
    return "\n".join(lines)

# --------------- Orchestration ---------------

def run_detection(url: str, token: str, org: str, bucket: str, start: datetime, end: datetime, metrics: List[str]) -> AnomalyReport:
    client = InfluxDBClient(url=url, token=token, org=org)
    try:
        df = query_metric_frames(client, bucket, org, start, end, metrics)
        stats = compute_baselines(df) if not df.empty else pd.DataFrame(columns=["metric","device","mean","std"])
        anomalies = detect_anomalies(df, stats)
        return AnomalyReport(start=start, end=end, generated_at=datetime.now(timezone.utc), anomalies=anomalies)
    finally:
        client.close()

# --------------- CLI ---------------

def parse_args():
    p = argparse.ArgumentParser(description="Detect anomalies in router metrics and build an LLM prompt.")
    p.add_argument("--url", default=os.getenv("INFLUXDB_URL","http://localhost:8086"))
    p.add_argument("--token", default=os.getenv("INFLUXDB_TOKEN",""))
    p.add_argument("--org", default=os.getenv("INFLUXDB_ORG",""))
    p.add_argument("--bucket", default=os.getenv("INFLUXDB_BUCKET","network_data"))
    p.add_argument("--start", required=False, default=None, help="UTC start time YYYY-MM-DD HH:MM:SS (default: now-24h)")
    p.add_argument("--end", required=False, default=None, help="UTC end time YYYY-MM-DD HH:MM:SS (default: now)")
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    p.add_argument("--llm-prompt", help="Write constructed LLM prompt to file")
    p.add_argument("--json", default="anomalies.json", help="Write JSON anomaly report to file (default: anomalies.json)")
    return p.parse_args()

def main():
    args = parse_args()
    now = datetime.now(timezone.utc)
    # Preflight: require token and org explicitly
    if not args.token:
        print("ERROR: Missing InfluxDB token. Set INFLUXDB_TOKEN env or pass --token.")
        return
    if not args.org:
        print("ERROR: Missing InfluxDB org. Set INFLUXDB_ORG env or pass --org.")
        return
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        end = now
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        start = end - pd.Timedelta(hours=24)
    try:
        report = run_detection(args.url, args.token, args.org, args.bucket, start, end, args.metrics)
    except ApiException as e:
        if e.status == 401:
            print("ERROR: Unauthorized (401). Check: --token value, org, bucket, and that token has read access.")
            print("Provided params summary (token redacted):")
            print(f"  url={args.url} org={args.org or '(empty)'} bucket={args.bucket} token_len={len(args.token) if args.token else 0}")
            print("Troubleshooting: 1) Verify INFLUXDB_TOKEN env. 2) In UI: load Data > API Tokens ensure read for bucket. 3) If bucket recreated, token still valid (tokens are not bucket-recreated). 4) Try passing --org explicitly.")
            return
        raise
    print(f"Detected {len(report.anomalies)} anomalies in window.")
    for a in report.anomalies[:20]:  # show sample
        print(f"{a.time.isoformat()} {a.device} {a.metric} {a.severity} value={a.value:.2f} z={a.zscore:.2f} -> {a.reason}")
    prompt = build_llm_prompt(report)
    if args.llm_prompt:
        with open(args.llm_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"LLM prompt written to {args.llm_prompt}")
    if args.json:
        import json
        # Convert dataclass anomalies ensuring datetime is ISO formatted
        anomalies_payload = []
        for a in report.anomalies:
            d = asdict(a)
            # Replace datetime with ISO8601 string
            if isinstance(a.time, datetime):
                d["time"] = a.time.isoformat()
            anomalies_payload.append(d)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "start": report.start.isoformat(),
                "end": report.end.isoformat(),
                "generated_at": report.generated_at.isoformat(),
                "anomalies": anomalies_payload
            }, f, indent=2)
        print(f"JSON report written to {args.json}")
    # Print tail of prompt for user preview
    print("\n--- LLM Prompt Preview (first 40 lines) ---")
    for i, line in enumerate(prompt.splitlines()[:40]):
        print(line)

if __name__ == "__main__":
    main()
