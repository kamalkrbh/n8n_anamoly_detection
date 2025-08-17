"""LLM-oriented analyzer for network anomalies.

This module loads anomalies detected by anomaly_detector.py, groups them into
incidents, infers likely causes, and suggests remediation. It can also build
provider-agnostic prompts for an LLM. By default, it performs heuristic
analysis locally and does NOT call any external LLM.

Usage examples:
  python llm_analyzer.py --input anomalies.json --output-json incidents.json --output-md analysis.md
  python llm_analyzer.py --input anomalies.json --window-minutes 10

Optional (scaffold only; no network calls unless you implement providers):
  python llm_analyzer.py --input anomalies.json --provider none  # default

"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error


# ---------------- LLM helpers ----------------

def default_model_for(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return ""


def build_overall_prompt(anomalies: List[Anomaly], incidents: List[Incident]) -> str:
    sev_counts: Dict[str, int] = {}
    for a in anomalies:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1
    lines: List[str] = [
        "You are a senior NOC/SRE. Analyze the following network anomalies and grouped incidents.",
        f"Total anomalies: {len(anomalies)} | Severity: {sev_counts}",
        f"Incidents grouped: {len(incidents)}",
        "\nSummaries:",
    ]
    for inc in incidents[:10]:
        lines.append(
            f"- {inc.id} sev={inc.severity} devices={','.join(inc.devices)} metrics={','.join(inc.metrics)} "
            f"window={inc.start.isoformat()}..{inc.end.isoformat()} priority={inc.priority_score}"
        )
    lines.append("\nRepresentative anomalies (first 30):")
    for a in anomalies[:30]:
        lines.append(
            f"- {a.time.isoformat()} {a.device} {a.metric} {a.severity} value={a.value:.2f} z={a.zscore:.2f} reason={a.reason}"
        )
    lines.append(
        "\nTasks: 1) Correlate incidents and propose root causes. 2) Estimate impact and urgency. "
        "3) Provide prioritized remediation steps. 4) List any data to collect next if uncertain."
    )
    return "\n".join(lines)


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        return json.loads(body.decode("utf-8"))


def call_openai(api_key: str, model: str, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a concise, expert network incident analyst."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = _http_post_json(url, headers, payload)
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(resp, indent=2)


def call_groq(api_key: str, model: str, prompt: str) -> str:
    # Groq provides an OpenAI-compatible endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a concise, expert network incident analyst."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = _http_post_json(url, headers, payload)
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(resp, indent=2)


def call_ollama(host: str, model: str, prompt: str) -> str:
    # Use /api/generate for simplicity (non-streaming)
    url = host.rstrip("/") + "/api/generate"
    headers = {}
    payload = {
        "model": model or "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    resp = _http_post_json(url, headers, payload)
    try:
        return resp.get("response", "").strip() or json.dumps(resp, indent=2)
    except Exception:
        return json.dumps(resp, indent=2)


# ---------------- Data types ----------------

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
class Incident:
    id: str
    devices: List[str]
    start: datetime
    end: datetime
    severity: str
    anomalies: List[Anomaly]
    metrics: List[str]
    summary: Optional[str] = None
    hypotheses: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None
    priority_score: Optional[int] = None


# ---------------- I/O helpers ----------------

def parse_time(ts: str) -> datetime:
    # Support ISO 8601 with timezone
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_anomalies(path: str) -> List[Anomaly]:
    # Support both JSON array/object and JSONL (one anomaly per line) for watch mode.
    anomalies_raw: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        anomalies_raw.append(obj)
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anomalies_raw = data.get("anomalies", []) if isinstance(data, dict) else data
    out: List[Anomaly] = []
    for a in anomalies_raw:
        try:
            raw_time = a.get("time")
            if isinstance(raw_time, str):
                try:
                    parsed_time = parse_time(raw_time)
                except Exception:
                    parsed_time = datetime.now(timezone.utc)
            elif isinstance(raw_time, datetime):
                parsed_time = raw_time
            else:
                parsed_time = datetime.now(timezone.utc)
            out.append(Anomaly(
                metric=a["metric"],
                device=a["device"],
                time=parsed_time,
                severity=a.get("severity", "minor"),
                reason=a.get("reason", ""),
                value=float(a.get("value", 0.0)),
                baseline_mean=float(a.get("baseline_mean", 0.0)),
                baseline_std=float(a.get("baseline_std", 0.0)),
                zscore=float(a.get("zscore", 0.0)),
                context=a.get("context", {}) or {},
            ))
        except Exception:
            # Skip malformed anomaly entries
            continue
    # Sort by time
    out.sort(key=lambda x: x.time)
    return out


# ---------------- Grouping into incidents ----------------

SEVERITY_RANK = {"minor": 1, "major": 2, "critical": 3}


def rank_to_sev(rank: int) -> str:
    for k, v in SEVERITY_RANK.items():
        if v == rank:
            return k
    return "minor"


def max_severity(a: Optional[str], b: Optional[str]) -> str:
    if not a:
        return b or "minor"
    if not b:
        return a
    return a if SEVERITY_RANK[a] >= SEVERITY_RANK[b] else b


def group_anomalies(anomalies: List[Anomaly], window_minutes: int = 5) -> List[Incident]:
    if not anomalies:
        return []
    window = timedelta(minutes=window_minutes)
    incidents: List[Incident] = []
    cur_group: List[Anomaly] = []
    cur_start: Optional[datetime] = None
    cur_end: Optional[datetime] = None
    cur_sev: Optional[str] = None

    def flush_group():
        nonlocal cur_group, cur_start, cur_end, cur_sev
        if not cur_group:
            return
        devices = sorted({a.device for a in cur_group})
        metrics = sorted({a.metric for a in cur_group})
        inc = Incident(
            id=f"inc-{len(incidents)+1}",
            devices=devices,
            start=cur_start or cur_group[0].time,
            end=cur_end or cur_group[-1].time,
            severity=cur_sev or "minor",
            anomalies=list(cur_group),
            metrics=metrics,
        )
        incidents.append(inc)
        # Reset
        cur_group = []
        cur_start = None
        cur_end = None
        cur_sev = None

    # Simple temporal grouping across all devices; contiguous events within window
    for a in anomalies:
        if not cur_group:
            cur_group = [a]
            cur_start = a.time
            cur_end = a.time
            cur_sev = a.severity
            continue
        # If within window of current end, group; otherwise flush
        if a.time - (cur_end or a.time) <= window:
            cur_group.append(a)
            cur_end = a.time
            cur_sev = max_severity(cur_sev, a.severity)
        else:
            flush_group()
            cur_group = [a]
            cur_start = a.time
            cur_end = a.time
            cur_sev = a.severity
    flush_group()
    return incidents


# ---------------- Heuristic analysis ----------------

ROOT_RULES = [
    {
        "name": "Network congestion / saturation",
        "when_any": [
            ("latency_ms", ">", 150),
            ("packet_loss", ">=", 2),
        ],
        "also_with": [("if_octets_out", ">=", 50_000_000)],
        "evidence": "High latency/loss alongside heavy outbound traffic",
        "remediation": [
            "Check interface utilization and queue drops on egress",
            "Consider QoS or rate-limiting heavy flows",
            "Temporarily shift/balance traffic or increase capacity",
        ],
    },
    {
        "name": "Inbound surge / data transfer spike",
        "when_any": [("if_octets_in", ">=", 50_000_000)],
        "also_with": [("connections_active", ">=", 5000)],
        "evidence": "Large inbound volume, often with connection storm",
        "remediation": [
            "Identify top talkers and traffic sources",
            "Apply temporary rate-limits or block abusive sources",
            "Validate if a scheduled job/backup is running",
        ],
    },
    {
        "name": "Device CPU saturation",
        "when_any": [("cpu_utilization", ">=", 90)],
        "also_with": [],
        "evidence": "Sustained CPU >= 90%",
        "remediation": [
            "Inspect processes/features consuming CPU",
            "Reduce control-plane load or enable hardware offload",
            "Consider upgrading hardware or redistributing traffic",
        ],
    },
    {
        "name": "Connection surge / potential overload",
        "when_any": [("connections_active", ">=", 5000)],
        "also_with": [("latency_ms", ">=", 150)],
        "evidence": "High connection count correlating with latency",
        "remediation": [
            "Scale application instances or connection pools",
            "Tune timeouts/keep-alives; shed excess load",
            "Investigate client behavior or bot surges",
        ],
    },
    {
        "name": "Packet loss without bandwidth spike",
        "when_any": [("packet_loss", ">=", 2)],
        "also_with": [],
        "evidence": "Loss observed even without extreme traffic",
        "remediation": [
            "Check interface errors, flaps, CRCs",
            "Verify link quality and duplex/MTU settings",
            "Trace path to identify external segment issues",
        ],
    },
]


def incident_signals(incident: Incident) -> Dict[str, float]:
    """Aggregate strongest value per metric within incident."""
    strongest: Dict[str, float] = {}
    for a in incident.anomalies:
        strongest[a.metric] = max(strongest.get(a.metric, float("-inf")), a.value)
    return strongest


def check_condition(value: Optional[float], op: str, threshold: float) -> bool:
    if value is None:
        return False
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    return False


def analyze_incident(incident: Incident) -> Incident:
    signals = incident_signals(incident)
    hyps: List[Dict[str, Any]] = []
    recs: List[str] = []

    # Rule-based hypotheses
    for rule in ROOT_RULES:
        fired_any = any(check_condition(signals.get(m), op, thr) for m, op, thr in rule["when_any"])
        if not fired_any:
            continue
        also_ok = all(check_condition(signals.get(m), op, thr) for m, op, thr in rule.get("also_with", []))
        confidence = 0.6 if also_ok else 0.4
        hyps.append({
            "name": rule["name"],
            "confidence": confidence,
            "evidence": rule["evidence"],
        })
        for r in rule["remediation"]:
            if r not in recs:
                recs.append(r)

    # Context: nighttime adjustments
    night_hits = 0
    for a in incident.anomalies:
        hour = a.context.get("hour")
        if isinstance(hour, int) and (hour in {23,0,1,2,3,4,5}):
            night_hits += 1
    if night_hits and any(m in incident.metrics for m in ("if_octets_in", "if_octets_out")):
        hyps.append({
            "name": "Unusual night traffic",
            "confidence": 0.4,
            "evidence": f"{night_hits} anomalies occurred during night hours",
        })
        maybe = "Verify scheduled backups/jobs or off-peak tasks"
        if maybe not in recs:
            recs.append(maybe)

    # Priority scoring
    sev_score = {"minor": 40, "major": 70, "critical": 100}[incident.severity]
    multi_metric_bonus = 10 if len(incident.metrics) >= 2 else 0
    loss_bonus = 15 if "packet_loss" in incident.metrics else 0
    duration_bonus = min(20, int((incident.end - incident.start).total_seconds() // 60))
    priority = sev_score + multi_metric_bonus + loss_bonus + duration_bonus

    # Summary text
    summary = (
        f"Incident {incident.id} on {','.join(incident.devices)} from {incident.start.isoformat()} "
        f"to {incident.end.isoformat()} | sev={incident.severity} | metrics={','.join(incident.metrics)}"
    )

    incident.hypotheses = hyps or [{"name": "Unclassified anomaly pattern", "confidence": 0.2, "evidence": "Insufficient signals"}]
    incident.recommendations = recs or ["Collect additional telemetry (logs/traces), then re-evaluate"]
    incident.priority_score = priority
    incident.summary = summary
    return incident


# ---------------- Prompt building ----------------

def build_prompt_for_incident(incident: Incident) -> str:
    lines = [
        "You are a senior NOC SRE. Analyze this incident and provide root-cause hypotheses, impact, and remediation.",
        f"Incident: {incident.id}",
        f"Devices: {', '.join(incident.devices)}",
        f"Window: {incident.start.isoformat()} to {incident.end.isoformat()} UTC",
        f"Severity: {incident.severity}",
        f"Metrics involved: {', '.join(incident.metrics)}",
        "Anomalies (time device metric severity value zscore reason):",
    ]
    for a in incident.anomalies[:100]:
        lines.append(f"- {a.time.isoformat()} {a.device} {a.metric} {a.severity} value={a.value:.2f} z={a.zscore:.2f} reason={a.reason}")
    if incident.hypotheses:
        lines.append("\nCurrent hypotheses (preliminary):")
        for h in incident.hypotheses[:10]:
            lines.append(f"- {h['name']} (confidence {h.get('confidence', 0):.0%}): {h.get('evidence','')}")
    lines.append(
        "\nTasks: 1) Confirm or refine hypotheses. 2) Assess user/business impact. 3) Propose prioritized remediation with steps. 4) Suggest next data to collect if uncertain."
    )
    return "\n".join(lines)


# ---------------- Serialization ----------------

def incident_to_dict(incident: Incident) -> Dict[str, Any]:
    def dt(d: datetime) -> str:
        return d.isoformat()
    return {
        "id": incident.id,
        "devices": incident.devices,
        "start": dt(incident.start),
        "end": dt(incident.end),
        "severity": incident.severity,
        "priority_score": incident.priority_score,
        "metrics": incident.metrics,
        "summary": incident.summary,
        "hypotheses": incident.hypotheses,
        "recommendations": incident.recommendations,
        "anomalies": [
            {
                "metric": a.metric,
                "device": a.device,
                "time": a.time.isoformat(),
                "severity": a.severity,
                "reason": a.reason,
                "value": a.value,
                "zscore": a.zscore,
                "baseline_mean": a.baseline_mean,
                "baseline_std": a.baseline_std,
                "context": a.context,
            }
            for a in incident.anomalies
        ],
        "prompt": build_prompt_for_incident(incident),
    }


# ---------------- CLI ----------------

from types import SimpleNamespace

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1","true","yes","on"}

def load_config():
    return SimpleNamespace(
        input=os.getenv("ANALYZER_INPUT","anomalies.json"),
        window_minutes=int(os.getenv("ANALYZER_WINDOW_MINUTES","5")),
        output_json=os.getenv("ANALYZER_OUTPUT_JSON","incidents.json"),
        output_md=os.getenv("ANALYZER_OUTPUT_MD","analysis.md"),
        provider=os.getenv("ANALYZER_PROVIDER","none"),
        model=os.getenv("ANALYZER_MODEL"),
        invoke_llm=env_bool("ANALYZER_INVOKE_LLM", False),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        ollama_host=os.getenv("OLLAMA_HOST","http://localhost:11434"),
        llm_out=os.getenv("ANALYZER_LLM_OUT","llm_response.txt"),
        watch=env_bool("ANALYZER_WATCH", False),
        tail_interval=float(os.getenv("ANALYZER_TAIL_INTERVAL","30")),
        llm_cooldown_seconds=int(os.getenv("ANALYZER_LLM_COOLDOWN_SECONDS","300")),
        min_trigger_severity=os.getenv("ANALYZER_MIN_TRIGGER_SEVERITY","major")
    )

SEV_RANK = {"minor":1,"major":2,"critical":3}

def watch_loop(args):
    path = args.input
    if not path.endswith('.jsonl'):
        print(f"[watch] Input {path} is not .jsonl; exiting watch mode.")
        return
    offset = 0
    last_llm_ts: Optional[datetime] = None
    min_rank = SEV_RANK.get(args.min_trigger_severity,2)
    print(f"[watch] tailing {path} interval={args.tail_interval}s min_trigger_sev={args.min_trigger_severity}")
    while True:
        try:
            sz = os.path.getsize(path)
        except OSError:
            time.sleep(args.tail_interval)
            continue
        if sz < offset:  # file rotated
            offset = 0
        new_anomalies: List[Anomaly] = []
        try:
            with open(path,'r',encoding='utf-8') as f:
                f.seek(offset)
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # minimal validation
                            if 'metric' in obj and 'device' in obj and 'time' in obj:
                                # reuse load logic quickly
                                new_anomalies.append(load_anomalies_from_list([obj])[0])
                    except Exception:
                        continue
                offset = f.tell()
        except Exception as e:
            print(f"[watch] read error: {e}")
        if new_anomalies:
            # Group with just new anomalies for immediate context or combine recent window: reload last window_minutes from file for richness
            # Simple: combine all anomalies loaded so far (could optimize)
            all_anoms = load_anomalies(path)
            incidents = group_anomalies(all_anoms[-1000:], window_minutes=args.window_minutes)  # limit history slice
            analyzed = [analyze_incident(i) for i in incidents]
            highest_rank = max((SEV_RANK.get(a.severity,1) for a in new_anomalies), default=1)
            now_ts = datetime.now(timezone.utc)
            should_llm = args.invoke_llm and args.provider != 'none' and (
                highest_rank >= min_rank or (last_llm_ts is None) or (now_ts - last_llm_ts).total_seconds() >= args.llm_cooldown_seconds
            )
            if should_llm:
                prompt = build_overall_prompt(all_anoms, analyzed)
                model = args.model or default_model_for(args.provider)
                try:
                    if args.provider == 'openai':
                        api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY','')
                        if not api_key: raise RuntimeError('Missing OpenAI API key.')
                        output = call_openai(api_key, model, prompt)
                    elif args.provider == 'groq':
                        api_key = args.groq_api_key or os.getenv('GROQ_API_KEY','')
                        if not api_key: raise RuntimeError('Missing Groq API key.')
                        output = call_groq(api_key, model, prompt)
                    else:
                        output = call_ollama(args.ollama_host, model, prompt)
                    last_llm_ts = now_ts
                    with open(args.llm_out, 'w', encoding='utf-8') as f:
                        f.write(output)
                    print(f"[watch][LLM] response written ({len(output.split())} words)")
                    print(output.splitlines()[0:40])
                except Exception as e:
                    print(f"[watch][LLM] failed: {e}")
            else:
                print(f"[watch] {len(new_anomalies)} new anomalies, no LLM trigger (rank={highest_rank})")
        try:
            time.sleep(args.tail_interval)
        except KeyboardInterrupt:
            print('[watch] stopped')
            break

def load_anomalies_from_list(lst: List[dict]) -> List[Anomaly]:
    # helper for watch_loop to create anomalies from raw dicts
    out: List[Anomaly] = []
    for a in lst:
        try:
            raw_time = a.get('time')
            if isinstance(raw_time, str):
                try:
                    parsed_time = parse_time(raw_time)
                except Exception:
                    parsed_time = datetime.now(timezone.utc)
            elif isinstance(raw_time, datetime):
                parsed_time = raw_time
            else:
                parsed_time = datetime.now(timezone.utc)
            out.append(Anomaly(metric=a['metric'], device=a['device'], time=parsed_time, severity=a.get('severity','minor'), reason=a.get('reason',''), value=float(a.get('value',0.0)), baseline_mean=float(a.get('baseline_mean',0.0)), baseline_std=float(a.get('baseline_std',0.0)), zscore=float(a.get('zscore',0.0)), context=a.get('context',{}) or {}))
        except Exception:
            continue
    return out


def main():
    args = load_config()
    if args.watch:
        watch_loop(args)
        return
    anomalies = load_anomalies(args.input)
    if not anomalies:
        print("No anomalies to analyze.")
        return
    incidents = group_anomalies(anomalies, window_minutes=args.window_minutes)
    analyzed: List[Incident] = [analyze_incident(inc) for inc in incidents]

    # Write JSON
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(analyzed),
        "incidents": [incident_to_dict(x) for x in analyzed],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Incidents JSON written to {args.output_json}")

    # Write Markdown
    lines: List[str] = ["# Incident Analysis\n"]
    for inc in analyzed:
        lines.append(f"## {inc.id} | sev={inc.severity} | priority={inc.priority_score}\n")
        lines.append(f"Devices: {', '.join(inc.devices)}\n")
        lines.append(f"Window: {inc.start.isoformat()} — {inc.end.isoformat()} UTC\n")
        lines.append(f"Metrics: {', '.join(inc.metrics)}\n")
        lines.append("### Hypotheses\n")
        for h in (inc.hypotheses or []):
            lines.append(f"- {h['name']} (confidence {h.get('confidence', 0):.0%}) — {h.get('evidence','')}\n")
        lines.append("### Recommendations\n")
        for r in (inc.recommendations or []):
            lines.append(f"- {r}\n")
        lines.append("### Prompt\n")
        lines.append("""```\n""".strip()+"\n")
        lines.append(incident_to_dict(inc)["prompt"]+"\n")
        lines.append("""```\n""".strip()+"\n")
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report written to {args.output_md}")

    # Optionally call an LLM to produce a final narrative
    if args.invoke_llm and args.provider != "none":
        print(f"Invoking LLM provider: {args.provider}")
        prompt = build_overall_prompt(anomalies, analyzed)
        model = args.model or default_model_for(args.provider)
        try:
            if args.provider == "openai":
                api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    raise RuntimeError("Missing OpenAI API key. Use --openai-api-key or set OPENAI_API_KEY.")
                output = call_openai(api_key, model, prompt)
            elif args.provider == "groq":
                api_key = args.groq_api_key or os.getenv("GROQ_API_KEY", "")
                if not api_key:
                    raise RuntimeError("Missing Groq API key. Use --groq-api-key or set GROQ_API_KEY.")
                output = call_groq(api_key, model, prompt)
            else:  # ollama
                output = call_ollama(args.ollama_host, model, prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            return
        with open(args.llm_out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"LLM response written to {args.llm_out}")
        print("\n--- LLM Response (first 60 lines) ---")
        for line in output.splitlines()[:60]:
            print(line)


if __name__ == "__main__":
    main()
