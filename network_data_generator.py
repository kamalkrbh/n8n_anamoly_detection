import argparse
import os
from dotenv import load_dotenv
import random
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import WriteOptions
from influxdb_client.client.delete_api import DeleteApi

# Load environment variables from .env file
load_dotenv()
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "your-org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "network_data")

# Common network metrics
METRICS = ["bandwidth", "latency", "packet_loss"]

# Anomaly types
ANOMALY_TYPES = ["spike", "drop", "outage"]

def generate_data(start_time, end_time, freq, anomaly_rate):
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq).tz_convert('UTC')
    data = []
    for ts in timestamps:
        # Normal values
        bandwidth = np.random.normal(100, 10)  # Mbps
        latency = np.random.normal(20, 5)      # ms
        packet_loss = np.random.normal(0.5, 0.2)  # %
        # Inject anomaly
        anomaly = None
        if random.random() < anomaly_rate:
            anomaly = random.choice(ANOMALY_TYPES)
            if anomaly == "spike":
                bandwidth *= random.uniform(2, 5)
                latency *= random.uniform(2, 5)
            elif anomaly == "drop":
                bandwidth *= random.uniform(0.1, 0.5)
                latency *= random.uniform(0.1, 0.5)
            elif anomaly == "outage":
                bandwidth = 0
                latency = 0
                packet_loss = 100
        data.append({
            "time": ts.to_pydatetime(),
            "bandwidth": float(round(bandwidth, 2)),
            "latency": float(round(latency, 2)),
            "packet_loss": float(round(packet_loss, 2)),
            "anomaly": anomaly or "normal"
        })
    return data

def write_to_influxdb(data):
    # ...existing code...
    
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    # Clean up previous data in 'network_metrics'
    # Drop and recreate the bucket for a full cleanup
    from influxdb_client.client.bucket_api import BucketsApi
    from influxdb_client.client.write_api import WriteType
    buckets_api = BucketsApi(client)
    # Find the bucket
    buckets = buckets_api.find_buckets().buckets
    bucket_obj = next((b for b in buckets if b.name == INFLUXDB_BUCKET), None)
    if bucket_obj:
        buckets_api.delete_bucket(bucket_obj)
        print(f"Bucket '{INFLUXDB_BUCKET}' deleted.")
    buckets_api.create_bucket(bucket_name=INFLUXDB_BUCKET, org=INFLUXDB_ORG)
    print(f"Bucket '{INFLUXDB_BUCKET}' recreated.")
    write_api = client.write_api(write_options=WriteOptions(write_type=WriteType.synchronous))
    try:
        timestamps_seen = set()
        for i, row in enumerate(data):
            ts = row["time"]
            # Ensure unique timestamp by adding microseconds if duplicate detected
            while ts in timestamps_seen:
                ts = ts + timedelta(microseconds=1)
            timestamps_seen.add(ts)
            print(f"Writing point {i}: time={ts}, anomaly={row['anomaly']}")
            point = (Point("network_metrics")
                     .field("bandwidth", row["bandwidth"]) \
                     .field("latency", row["latency"]) \
                     .field("packet_loss", row["packet_loss"]) \
                     .tag("anomaly", row["anomaly"]))
            # Use the generated timestamp for EVERY point (previously first point was overwritten -> caused mismatch)
            point = point.time(ts)
            write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        # Debug: show written time bounds
        written_times_sorted = sorted(timestamps_seen)
        print(f"Written time range: {written_times_sorted[0]} -> {written_times_sorted[-1]} (total unique timestamps: {len(written_times_sorted)})")
        write_api.flush()
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")
    finally:
        client.close()
    # In-function verification after close (separate client)
    client_v = None
    try:
        client_v = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        q = client_v.query_api()
        start_iso = data[0]["time"].isoformat()
        end_exclusive_iso = (data[-1]["time"] + timedelta(microseconds=1)).isoformat()
        flux_verify = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {start_iso}, stop: {end_exclusive_iso})
    |> filter(fn: (r) => r._measurement == "network_metrics")
    |> filter(fn: (r) => r._field == "bandwidth")
    |> group(columns: [])
    |> count()
'''
        res = q.query(org=INFLUXDB_ORG, query=flux_verify)
        total = 0
        for tbl in res:
            for rec in tbl.records:
                total += int(rec.get_value())
        print(f"(Internal) Bandwidth field points counted: {total} vs expected {len(data)}")
    except Exception as ve:
        print(f"Internal verification error: {ve}")
    finally:
        try:
            client_v.close()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Generate and push network time series data to InfluxDB.")
    parser.add_argument("--start", type=str, default=None, help="Start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", type=str, default=None, help="End time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--freq", type=str, default="1min", help="Frequency of data points (e.g., '1min', '5s')")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="Probability of anomaly per data point")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if args.start else now - timedelta(hours=1)
    end_time = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if args.end else now

    print(f"Generating data from {start_time} to {end_time} every {args.freq} with anomaly rate {args.anomaly_rate}")
    data = generate_data(start_time, end_time, args.freq, args.anomaly_rate)
    print(f"Generated {len(data)} data points. Writing to InfluxDB...")
    write_to_influxdb(data)

    # Verification: count logical data points (one per timestamp) by counting a single field (bandwidth)
    client = None
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        query_api = client.query_api()
        end_exclusive_main = (data[-1]["time"] + timedelta(microseconds=1)).isoformat()
        flux = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {data[0]["time"].isoformat()}, stop: {end_exclusive_main})
    |> filter(fn: (r) => r._measurement == "network_metrics")
    |> filter(fn: (r) => r._field == "bandwidth")
    |> group(columns: [])
    |> count()
        '''
        result = query_api.query(org=INFLUXDB_ORG, query=flux)
        total = 0
        for table in result:
            for record in table.records:
                # _value holds the count after count()
                total += int(record.get_value())
        print(f"Verified logical data points (by 'bandwidth' field): {total}. Expected: {len(data)}")
        if total == len(data):
            print("SUCCESS: Point count matches.")
        else:
            print("WARNING: Mismatch between written logical points and expected count.")
    except Exception as e:
        print(f"Verification error: {e}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    print("Done.")

if __name__ == "__main__":
    main()
