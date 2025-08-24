import os
import math
from datetime import datetime, timedelta, timezone
import time
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import WriteOptions, SYNCHRONOUS

load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "your-org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "network_data")

DEFAULT_ANOMALY_PROB = float(os.getenv("GEN_ANOMALY_PROB", os.getenv("ANOMALY_PROB", 0.005)))
DEFAULT_NOISE_LEVEL = float(os.getenv("GEN_NOISE_LEVEL", os.getenv("NOISE_LEVEL", 1.0)))
DEFAULT_NUM_DEVICES = int(os.getenv("GEN_DEVICES", os.getenv("NUM_DEVICES", 3)))
DEFAULT_DEVICE_PREFIX = os.getenv("GEN_DEVICE_PREFIX", os.getenv("DEVICE_PREFIX", "router"))
DEFAULT_WRITE_MODE = os.getenv("GEN_WRITE_MODE", os.getenv("WRITE_MODE", "sync"))
DEFAULT_START = os.getenv("GEN_START", os.getenv("START", None))
DEFAULT_END = os.getenv("GEN_END", os.getenv("END", None))
DEFAULT_SEED = os.getenv("GEN_SEED", os.getenv("SEED", None))
RECREATE_BUCKET_FLAG = os.getenv("GEN_RECREATE_BUCKET", os.getenv("RECREATE_BUCKET", "true")).lower() in ("1","true","yes","on")
CONTINUOUS_FLAG = os.getenv("GEN_CONTINUOUS", os.getenv("CONTINUOUS", "false")).lower() in ("1","true","yes","on")
CONTINUOUS_INTERVAL = int(os.getenv("GEN_INTERVAL_SECONDS", os.getenv("INTERVAL_SECONDS", 60)))
CONTINUOUS_ANOMALY_STATE_MINUTES = int(os.getenv("GEN_ANOMALY_STATE_MINUTES", 30))  # how long we keep sustained anomaly windows

# Device IP address configuration
IP_PREFIX = os.getenv("GEN_IP_PREFIX", os.getenv("IP_PREFIX", "10.0.0."))
IP_START = int(os.getenv("GEN_IP_START", os.getenv("IP_START", 1)))

METRICS = [
    "if_octets_in",
    "if_octets_out",
    "cpu_utilization",
    "packet_loss",
    "latency_ms",
    "connections_active"
]

def get_metric_unit(metric: str) -> str:
    if metric in ("if_octets_in", "if_octets_out", "connections_active"):
        return "count"
    if metric == "cpu_utilization":
        return "%"
    if metric == "packet_loss":
        return "%"
    if metric == "latency_ms":
        return "ms"
    return "unknown"

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
        # Raise baseline and variance so normal 40–60% values aren't flagged easily
        base = 40 + 20 * (cycle + 1) / 2  # was 30 + 15 * ...
        noise = rng.normal(0, 5 * noise_level, size=len(ts))  # was 2 * noise_level
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

def build_points(metric: str, values: np.ndarray, ts, device: str, ip: str) -> List[Point]:
    pts = []
    unit = get_metric_unit(metric)
    for v, t in zip(values, ts):
        pts.append(
            Point(metric)
            .tag("device", device)
            .tag("ip", ip)
            .field("value", round(float(v), 2))
            .field("unit", unit)
            .time(t.to_pydatetime())
        )
    return pts

def write_metrics(devices: List[str], device_ips: dict[str, str], ts, anomaly_prob: float, noise_level: float, seed: int | None, write_mode: str = "sync"):
    rng = np.random.default_rng(seed)
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    if write_mode == "async":
        write_api = client.write_api(write_options=WriteOptions(batch_size=5000, flush_interval=10_000, jitter_interval=2_000, retry_interval=5_000))
    else:
        write_api = client.write_api(write_options=SYNCHRONOUS)
    total_points = 0
    try:
        for device in devices:
            ip = device_ips.get(device, "")
            print(f"Generating metrics for {device} ...")
            for metric in METRICS:
                vals = metric_baseline(metric, ts, noise_level, rng)
                apply_anomalies(metric, vals, ts, anomaly_prob, rng)
                pts = build_points(metric, vals, ts, device, ip)
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

def main():
    # Wait for InfluxDB to be reachable (avoid ConnectionRefusedError on cold start)
    def wait_for_influx(timeout_seconds: int = 180, interval_seconds: int = 3):
        import urllib.request
        url = INFLUXDB_URL.rstrip('/') + '/health'
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        print(f"InfluxDB is healthy at {url}")
                        return True
            except Exception:
                pass
            print(f"Waiting for InfluxDB at {url} ...")
            time.sleep(interval_seconds)
        print("WARN: Timeout waiting for InfluxDB health; proceeding and will rely on client retries if any.")
        return False

    # Read configuration exclusively from environment variables
    anomaly_prob = DEFAULT_ANOMALY_PROB
    noise_level = DEFAULT_NOISE_LEVEL
    devices_count = DEFAULT_NUM_DEVICES
    device_prefix = DEFAULT_DEVICE_PREFIX
    write_mode = DEFAULT_WRITE_MODE if DEFAULT_WRITE_MODE in ("sync","async") else "sync"
    seed: Optional[int] = None
    if DEFAULT_SEED is not None:
        try:
            seed = int(DEFAULT_SEED)
        except ValueError:
            print(f"WARN: Invalid GEN_SEED/SEED value '{DEFAULT_SEED}' - ignoring")
    start_env = DEFAULT_START
    end_env = DEFAULT_END
    end = datetime.strptime(end_env, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if end_env else datetime.now(timezone.utc)
    start = datetime.strptime(start_env, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if start_env else end - timedelta(days=7)
    # Ensure DB is up before any API calls
    wait_for_influx()
    if RECREATE_BUCKET_FLAG:
        recreate_bucket()
    ts = generate_timestamps(start, end)
    devices = [f"{device_prefix}{i+1}" for i in range(devices_count)]
    # Build deterministic IP addresses per device (e.g., 10.0.0.1, 10.0.0.2, ...)
    device_ips = {dev: f"{IP_PREFIX}{IP_START + idx}" for idx, dev in enumerate(devices)}
    print("Device IP map:")
    for d, ip in device_ips.items():
        print(f"  {d} -> {ip}")
    print("Configuration:")
    print(f"  anomaly_prob={anomaly_prob} noise_level={noise_level} devices={devices_count} prefix={device_prefix} write_mode={write_mode} seed={seed}")
    print(f"  window_start={start.isoformat()} window_end={end.isoformat()} recreate_bucket={RECREATE_BUCKET_FLAG} continuous={CONTINUOUS_FLAG} interval={CONTINUOUS_INTERVAL}s")
    print(f"Generating initial history: {len(ts)} minutes ({(end-start)} range) for metrics: {', '.join(METRICS)} across {len(devices)} devices")
    write_metrics(devices, device_ips, ts, anomaly_prob, noise_level, seed, write_mode)
    if not CONTINUOUS_FLAG:
        print("Initial generation complete (one-shot mode). Exiting.")
        return
    # Continuous extension: tick every CONTINUOUS_INTERVAL seconds (no minute alignment)
    print("Switching to continuous mode...")
    next_ts = (end + timedelta(seconds=CONTINUOUS_INTERVAL)).replace(microsecond=0)
    rng = np.random.default_rng(seed)
    sustained_state = {}  # (device, metric) -> { 'kind': str, 'end': datetime }
    try:
        while True:
            now_aligned = datetime.now(timezone.utc).replace(microsecond=0)
            # If we're catching up from historical end, generate ticks until we reach current
            while next_ts <= now_aligned:
                cycle_index = next_ts.hour * 60 + next_ts.minute + (next_ts.second / 60.0)
                cycle_val = math.sin(2 * math.pi * (cycle_index % 1440) / 1440)
                points = []
                for device in devices:
                    ip = device_ips.get(device, "")
                    for metric in METRICS:
                        # Remove expired sustained anomaly
                        key = (device, metric)
                        if key in sustained_state and next_ts >= sustained_state[key]['end']:
                            del sustained_state[key]
                        # Baseline single value (reuse baseline formulas simplified)
                        if metric == "if_octets_in":
                            base = 5e6 + 3e6 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 2e5 * noise_level)
                        elif metric == "if_octets_out":
                            base = 4e6 + 2.5e6 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 2e5 * noise_level)
                        elif metric == "cpu_utilization":
                            # Match raised baseline/noise in continuous mode
                            base = 40 + 20 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 5 * noise_level)
                        elif metric == "packet_loss":
                            base = 0.2 + 0.05 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 0.05 * noise_level)
                        elif metric == "latency_ms":
                            base = 20 + 5 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 1.5 * noise_level)
                        elif metric == "connections_active":
                            base = 1000 + 400 * (cycle_val + 1) / 2
                            noise = rng.normal(0, 30 * noise_level)
                        else:
                            base = 0.0
                            noise = 0.0
                        val = base + noise
                        # Possibly start anomaly
                        if rng.random() < anomaly_prob:
                            if metric in ("if_octets_in", "if_octets_out", "connections_active"):
                                kind = rng.choice(["spike", "drop", "sustained_high"])
                            elif metric == "cpu_utilization":
                                kind = rng.choice(["spike", "drop", "sustained_high"])
                            elif metric == "packet_loss":
                                kind = rng.choice(["spike", "loss_burst"])
                            elif metric == "latency_ms":
                                kind = rng.choice(["spike", "latency_surge", "drop"])
                            else:
                                kind = "spike"
                            if kind in ("sustained_high", "latency_surge", "loss_burst"):
                                duration = int(rng.integers(3, 12))
                                sustained_state[key] = { 'kind': kind, 'end': next_ts + timedelta(minutes=duration) }
                            else:
                                # apply immediate single-point anomaly
                                val = apply_single_point_anomaly(kind, metric, val, rng)
                        # Apply sustained anomaly transformation if active
                        if key in sustained_state:
                            val = apply_sustained_anomaly(sustained_state[key]['kind'], metric, val, rng)
                        # clip
                        if metric in ("cpu_utilization", "packet_loss"):
                            val = max(0, min(100, val))
                        if metric in ("if_octets_in", "if_octets_out", "latency_ms", "connections_active"):
                            val = max(0, val)
                        points.append(
                            Point(metric)
                            .tag("device", device)
                            .tag("ip", ip)
                            .field("value", float(val))
                            .field("unit", get_metric_unit(metric))
                            .time(next_ts)
                        )
                # write tick batch
                client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
                w_api = client.write_api(write_options=SYNCHRONOUS)
                for i in range(0, len(points), 1000):
                    w_api.write(bucket=INFLUXDB_BUCKET, record=points[i:i+1000])
                client.close()
                print(f"[continuous] wrote {len(points)} points for {next_ts.isoformat()} active_sustained={len(sustained_state)}")
                next_ts += timedelta(seconds=CONTINUOUS_INTERVAL)
            # Sleep until the next scheduled tick to maintain ~fixed cadence
            sleep_seconds = (next_ts - datetime.now(timezone.utc)).total_seconds()
            if sleep_seconds > 0:
                time.sleep(min(CONTINUOUS_INTERVAL, sleep_seconds))
    except KeyboardInterrupt:
        print("Continuous mode stopped by user.")

def apply_single_point_anomaly(kind: str, metric: str, val: float, rng: np.random.Generator) -> float:
    if metric in ("if_octets_in", "if_octets_out", "connections_active"):
        if kind == "spike":
            return val * rng.uniform(3, 7)
        if kind == "drop":
            return val * rng.uniform(0.0, 0.2)
    if metric == "cpu_utilization":
        if kind == "spike":
            return min(100, val + rng.uniform(30, 55))
        if kind == "drop":
            return val * rng.uniform(0.1, 0.5)
    if metric == "packet_loss" and kind == "spike":
        return rng.uniform(5, 20)
    if metric == "latency_ms":
        if kind in ("spike", "latency_surge"):
            return val * rng.uniform(2, 3.5)
        if kind == "drop":
            return val * rng.uniform(0.3, 0.7)
    return val

def apply_sustained_anomaly(kind: str, metric: str, val: float, rng: np.random.Generator) -> float:
    if metric in ("if_octets_in", "if_octets_out", "connections_active") and kind == "sustained_high":
        return val * rng.uniform(1.5, 2.2)
    if metric == "cpu_utilization" and kind == "sustained_high":
        return min(100, val + rng.uniform(15, 30))
    if metric == "packet_loss" and kind == "loss_burst":
        return rng.uniform(5, 25)
    if metric == "latency_ms" and kind == "latency_surge":
        return val * rng.uniform(2, 3.2)
    return val

if __name__ == "__main__":
    main()
