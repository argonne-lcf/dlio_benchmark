#!/usr/bin/env python3
"""
Quick end-to-end test for the s3dlio ParquetRowGroupDataset sync iterator.

Usage:
    cd /home/eval/Documents/Code/dlio_benchmark
    source .venv/bin/activate
    python tests/test_s3dlio_parquet_loader.py

Expects:
  - s3-ultra running on port 9200 with minioadmin credentials
  - s3://mlp-flux/data/dlrm/train/ populated with 64 parquet files
"""

import os
import sys
import time

# ── S3 endpoint config ────────────────────────────────────────────────────────
os.environ.setdefault("AWS_ENDPOINT_URL_S3", "http://localhost:9200")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

import s3dlio  # noqa: E402

PREFIX = "s3://mlp-flux/data/dlrm/train/"
PREFETCH = 16


def run_s3_test():
    print(f"\n=== s3dlio ParquetRowGroupDataset sync iterator test ===")
    print(f"URI    : {PREFIX}")
    print(f"Prefetch: {PREFETCH}")

    loader = s3dlio.create_async_loader(
        PREFIX,
        {"format": "parquet", "prefetch": PREFETCH},
    )

    count = 0
    total_bytes = 0
    errors = 0

    t0 = time.perf_counter()
    for item in loader:
        try:
            n = len(item)  # PyBytesView implements __len__ via buffer protocol
            total_bytes += n
            count += 1
            if count % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate = total_bytes / elapsed / 1e9
                print(f"  {count} row groups | {total_bytes/1e9:.3f} GB | {rate:.3f} GB/s", flush=True)
        except Exception as e:
            errors += 1
            print(f"  ERROR at item {count}: {e}", file=sys.stderr)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Results ---")
    print(f"Row groups : {count}")
    print(f"Total bytes: {total_bytes / 1e9:.3f} GB")
    print(f"Elapsed    : {elapsed:.2f} s")
    if elapsed > 0:
        print(f"Throughput : {total_bytes / elapsed / 1e9:.3f} GB/s")
    if errors:
        print(f"Errors     : {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_s3_test()
