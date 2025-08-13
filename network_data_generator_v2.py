import os
import math
import argparse
from datetime import datetime, timedelta, timezone
import random
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import WriteOptions, SYNCHRONOUS

load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "your-org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "network_data")

DEFAULT_ANOMALY_PROB = 0.005
DEFAULT_NOISE_LEVEL = 1.0
DEFAULT_NUM_DEVICES = 3
DEFAULT_DEVICE_PREFIX = "router"

METRICS = [
    "if_octets_in",
    "if_octets_out",
    "cpu_utilization",
    "packet_loss",
    "latency_ms",
    "connections_active"
]

def generate_timestamps(start: datetime, end: datetime):
    return pd.date_range(start=start, end=end, freq="1min", tz=timezone.utc)

def daily_cycle(minute_index: np.ndarray, period_minutes: int = 1440) -> np.ndarray:
    return np.sin(2 * np.pi * (minute_index % period_minutes) / period_minutes)

def metric_baseline(metric: str, ts, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    minutes = np.arange(len(ts))
    cycle = daily_cycle(minutes)
    if metric == "if_octets_in":
        base = 5e6 + 3e6 * (cycle + 1) / 2
        noise = rng.normal(0, 2e5 * noise_level, size=len(ts))
    elif metric == "if_octets_out":
        base = 4e6 + 2.5e6 * (cycle + 1) / 2
        noise = rng.normal(0, 2e5 * noise_level, size=len(ts))
    elif metric == "cpu_utilization":
        base = 30 + 15 * (cycle + 1) / 2
        noise = rng.normal(0, 2 * noise_level, size=len(ts))
    elif metric == "packet_loss":
        base = np.full(len(ts), 0.2) + 0.05 * (cycle + 1) / 2
        noise = rng.normal(0, 0.05 * noise_level, size=len(ts))
    elif metric == "latency_ms":
        base = 20 + 5 * (cycle + 1) / 2
        noise = rng.normal(0, 1.5 * noise_level, size=len(ts))
    elif metric == "connections_active":
        base = 1000 + 400 * (cycle + 1) / 2
        noise = rng.normal(0, 30 * noise_level, size=len(ts))
    else:
        base = np.zeros(len(ts))
        noise = np.zeros(len(ts))
    return base + noise

def apply_anomalies(metric: str, values: np.ndarray, ts, anomaly_prob: float, rng: np.random.Generator):
    n = len(values)
    anomaly_indices = np.where(rng.random(n) < anomaly_prob)[0]
    for idx in anomaly_indices:
        kind = rng.choice(["spike", "drop", "sustained_high", "latency_surge", "loss_burst"])
        if metric in ("if_octets_in", "if_octets_out", "connections_active"):
            if kind == "spike":
                values[idx] *= rng.uniform(3, 8)
            elif kind == "drop":
                values[idx] *= rng.uniform(0.0, 0.2)
            elif kind == "sustained_high":
                span = rng.integers(5, 30)
                end = min(n, idx + span)
                values[idx:end] *= rng.uniform(1.5, 2.5)
        elif metric == "cpu_utilization":
            if kind == "spike":
                values[idx] += rng.uniform(30, 60)
            elif kind == "drop":
                values[idx] *= rng.uniform(0.1, 0.5)
            elif kind == "sustained_high":
                span = rng.integers(5, 30)
                end = min(n, idx + span)
                values[idx:end] += rng.uniform(20, 40)
        elif metric == "packet_loss":
            if kind in ("spike", "loss_burst"):
                span = rng.integers(2, 15)
                end = min(n, idx + span)
                values[idx:end] = rng.uniform(5, 25)
        elif metric == "latency_ms":
            if kind in ("spike", "latency_surge"):
                span = rng.integers(2, 20)
                end = min(n, idx + span)
                values[idx:end] *= rng.uniform(2, 4)
            elif kind == "drop":
                values[idx] *= rng.uniform(0.3, 0.7)
    if metric == "cpu_utilization" or metric == "packet_loss":
        values[:] = np.clip(values, 0, 100)
    if metric in ("if_octets_in", "if_octets_out", "latency_ms", "connections_active"):
        values[:] = np.clip(values, 0, None)

def build_points(metric: str, values: np.ndarray, ts, device: str) -> List[Point]:
    pts = []
    for v, t in zip(values, ts):
        pts.append(Point(metric).tag("device", device).field("value", float(v)).time(t.to_pydatetime()))
    return pts

def write_metrics(devices: List[str], ts, anomaly_prob: float, noise_level: float, seed: int | None, write_mode: str = "sync"):
    rng = np.random.default_rng(seed)
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    if write_mode == "async":
        write_api = client.write_api(write_options=WriteOptions(batch_size=5000, flush_interval=10_000, jitter_interval=2_000, retry_interval=5_000))
    else:
        write_api = client.write_api(write_options=SYNCHRONOUS)
    total_points = 0
    try:
        for device in devices:
            print(f"Generating metrics for {device} ...")
            for metric in METRICS:
                vals = metric_baseline(metric, ts, noise_level, rng)
                apply_anomalies(metric, vals, ts, anomaly_prob, rng)
                pts = build_points(metric, vals, ts, device)
                total_points += len(pts)
                for i in range(0, len(pts), 5000):
                    write_api.write(bucket=INFLUXDB_BUCKET, record=pts[i:i+5000])
            print(f"Completed {device}")
        if write_mode == "async":
            write_api.flush()
        print(f"Finished writing. Total points: {total_points}")
    finally:
        client.close()
    return total_points

def recreate_bucket():
    """Delete and recreate the target bucket so each run starts clean."""
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    try:
        buckets_api = client.buckets_api()
        existing = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET)
        if existing:
            print(f"Deleting existing bucket '{INFLUXDB_BUCKET}' (id={existing.id})")
            buckets_api.delete_bucket(existing)
        orgs_api = client.organizations_api()
        orgs = orgs_api.find_organizations(org=INFLUXDB_ORG)
        if not orgs:
            raise RuntimeError(f"Org '{INFLUXDB_ORG}' not found")
        org = orgs[0]
        print(f"Creating bucket '{INFLUXDB_BUCKET}' in org '{INFLUXDB_ORG}'")
        buckets_api.create_bucket(bucket_name=INFLUXDB_BUCKET, org_id=org.id)
    finally:
        client.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-metric network telemetry with synthetic anomalies.")
    parser.add_argument("--anomaly-prob", type=float, default=DEFAULT_ANOMALY_PROB, help="Per-point anomaly probability")
    parser.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL, help="Noise level scale")
    parser.add_argument("--devices", type=int, default=DEFAULT_NUM_DEVICES, help="Number of devices to simulate")
    parser.add_argument("--device-prefix", type=str, default=DEFAULT_DEVICE_PREFIX, help="Device name prefix")
    parser.add_argument("--start", type=str, default=None, help="Override start time (UTC, YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", type=str, default=None, help="Override end time (UTC, YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--write-mode", choices=["sync", "async"], default="sync", help="Write mode: sync (safe) or async (faster)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Always recreate bucket for a clean run
    recreate_bucket()
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if args.start else end - timedelta(days=7)
    ts = generate_timestamps(start, end)
    devices = [f"{args.device_prefix}{i+1}" for i in range(args.devices)]
    print(f"Generating {len(ts)} minutes ({(end-start)} range) for metrics: {', '.join(METRICS)} across {len(devices)} devices")
    total = write_metrics(devices, ts, args.anomaly_prob, args.noise_level, args.seed, args.write_mode)
    print("Done.")

if __name__ == "__main__":
    main()
