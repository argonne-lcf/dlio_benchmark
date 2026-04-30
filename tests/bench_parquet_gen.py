#!/usr/bin/env python3
"""bench_parquet_gen.py — Parquet data generation benchmark: BEFORE vs. AFTER.

Measures three things independently:

  SECTION 1 — Pure data generation (no Arrow, no disk)
    Both methods create a fresh Generator/Pool BEFORE timing starts.
    Only the actual data-generation call(s) are timed.

    BEFORE: Generator(size=N) created first → time get_chunk(N)
    AFTER small (<32 MB): BufferPool() created first → time next_slice(N)
    AFTER large (>=32 MB): Generator(size=N) created first → time fill_chunk loop

  SECTION 2 — Parquet encode to memory (no disk write)
    Generation + Arrow wrapping + encoding → pa.BufferOutputStream.
    Isolates encoding overhead from disk I/O.

  SECTION 3 — Full end-to-end to NVMe (/mnt/nvme_data)
    The real workload: generation + encoding + disk write.
    On this VM ~2 GB/s is the vSAN ceiling.

File sizes: 1 MB, 10 MB, 100 MB, 1 GB  (10 GB: manual, --include-10gb)
ELEM_SIZE:  512 KB/sample

NOTE on Generator state:
  Generator.current_pos is not reset by set_seed(), so a fully-drained
  Generator cannot be reused. The fix here: create a fresh Generator per rep
  (outside the timing window). dgen-py uses a random seed by default, which
  is fine — seeding is only needed when you want byte-identical runs.

NOTE on scaling:
  This benchmark runs in a single process and uses ~1 CPU core for Parquet
  encoding (PyArrow C++ encoder is serial per write_table call). In production,
  DLIO launches N MPI ranks each generating 1/N of the files in parallel, so
  real-world throughput scales linearly with rank count.

Usage
-----
  python tests/bench_parquet_gen.py            # sections 1-3, up to 1 GB
  python tests/bench_parquet_gen.py --gen-only # section 1 only (fastest)
  python tests/bench_parquet_gen.py --include-10gb
  python -m pytest tests/bench_parquet_gen.py -s -v
"""

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dgen_py

# ── Constants ─────────────────────────────────────────────────────────────────
ELEM_SIZE   = 512 * 1024        # 512 KB per sample
CHUNK_SIZE  = 64 * 1024 * 1024  # fill_chunk buffer (>=32 MB required by dgen-py)
POOL_THRESH = 32 * 1024 * 1024  # use BufferPool for totals/batches strictly below this
ROW_GROUP   = 64                # Parquet row-group size in samples
REPS        = 3                 # timed reps (+ 1 warm-up per section)

OUTPUT_DIR = os.environ.get("DLIO_BENCH_OUTPUT_DIR", "/mnt/nvme_data")

# (label, num_samples, include_in_default_run)
CASES = [
    ("1 MB",   2,      True),
    ("10 MB",  20,     True),
    ("100 MB", 200,    True),
    ("1 GB",   2048,   True),
    ("10 GB",  20480,  False),   # manual only: --include-10gb
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _human(n):
    for u, d in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= d:
            return f"{n/d:.0f} {u}"
    return f"{n} B"

def _tput(nbytes, secs):
    gbs = nbytes / secs / 1e9
    return f"{gbs:.2f} GB/s" if gbs >= 1.0 else f"{gbs*1000:.0f} MB/s"

def _schema(elem_size):
    return pa.schema([("data", pa.list_(pa.uint8(), list_size=elem_size))])

def _avg(lst):
    return sum(lst) / len(lst)


# ── SECTION 1: Pure data generation — no Arrow, no disk ──────────────────────
#
# Rule: Generator/Pool created BEFORE timing. Only fill call(s) are timed.
# Fresh Generator per rep because current_pos is never reset by set_seed().

def _gen_before_timed(file_bytes):
    """BEFORE: create Generator (not timed) -> time get_chunk only."""
    gen = dgen_py.Generator(size=file_bytes)  # NOT timed
    t0  = time.perf_counter()
    bv  = gen.get_chunk(file_bytes)
    t   = time.perf_counter() - t0
    del bv, gen
    return t

def _gen_after_pool_timed(file_bytes):
    """AFTER small file: create BufferPool (not timed) -> time next_slice only."""
    pool = dgen_py.BufferPool()               # NOT timed
    t0   = time.perf_counter()
    raw  = pool.next_slice(file_bytes)
    t    = time.perf_counter() - t0
    del raw, pool
    return t

def _gen_after_stream_timed(file_bytes, buf):
    """AFTER large file: create Generator (not timed) -> time fill_chunk loop."""
    gen = dgen_py.Generator(size=file_bytes)  # NOT timed
    t0  = time.perf_counter()
    while not gen.is_complete():
        gen.fill_chunk(buf)
    t = time.perf_counter() - t0
    del gen
    return t

def bench_gen_only(label, num_samples):
    file_bytes = num_samples * ELEM_SIZE
    use_pool   = file_bytes < POOL_THRESH
    buf        = None if use_pool else dgen_py.create_bytearrays(count=1, size=CHUNK_SIZE)[0]
    after_desc = "pool.next_slice  " if use_pool else f"fill_chunk({_human(CHUNK_SIZE)})"
    ram_after  = _human(file_bytes if use_pool else CHUNK_SIZE)

    print(f"\n  [{label}  {_human(file_bytes)}]")

    # BEFORE
    _gen_before_timed(file_bytes)                                      # warm-up
    times_b = [_gen_before_timed(file_bytes) for _ in range(REPS)]
    avg_b   = _avg(times_b)
    print(f"    BEFORE  get_chunk:      {' '.join(f'{t:.3f}s' for t in times_b)}"
          f"  -> avg {_tput(file_bytes, avg_b):>12}  peak RAM ~{_human(file_bytes)}")

    # AFTER
    if use_pool:
        _gen_after_pool_timed(file_bytes)                              # warm-up
        times_a = [_gen_after_pool_timed(file_bytes) for _ in range(REPS)]
    else:
        _gen_after_stream_timed(file_bytes, buf)                       # warm-up
        times_a = [_gen_after_stream_timed(file_bytes, buf) for _ in range(REPS)]
    avg_a = _avg(times_a)
    print(f"    AFTER   {after_desc}: {' '.join(f'{t:.3f}s' for t in times_a)}"
          f"  -> avg {_tput(file_bytes, avg_a):>12}  peak RAM ~{ram_after}")

    print(f"    Speedup: {avg_b/avg_a:.1f}x")


# ── SECTION 2: Parquet encode to memory (no disk) ─────────────────────────────

def _to_mem_before_timed(num_samples, elem_size):
    """BEFORE: Generator created (not timed) -> time get_chunk + Arrow + encode."""
    file_bytes  = num_samples * elem_size
    schema      = _schema(elem_size)
    num_batches = (num_samples + ROW_GROUP - 1) // ROW_GROUP

    gen = dgen_py.Generator(size=file_bytes)          # NOT timed

    t0   = time.perf_counter()
    bv   = gen.get_chunk(file_bytes)
    flat = np.frombuffer(bv, dtype=np.uint8).copy()
    del bv, gen

    arrow_flat = pa.array(flat, type=pa.uint8())
    arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
    full_table = pa.table({"data": arrow_data})
    del flat

    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(sink, schema, compression=None) as w:
        for bi in range(num_batches):
            start    = bi * ROW_GROUP
            cur_rows = min(ROW_GROUP, num_samples - start)
            w.write_table(full_table.slice(start, cur_rows), row_group_size=ROW_GROUP)

    encoded = sink.getvalue()
    return time.perf_counter() - t0, len(encoded)

def _to_mem_after_timed(num_samples, elem_size, buf):
    """AFTER: Generator/Pool created (not timed) -> time row-group loop + Arrow + encode."""
    file_bytes  = num_samples * elem_size
    batch_bytes = min(ROW_GROUP, num_samples) * elem_size
    schema      = _schema(elem_size)
    num_batches = (num_samples + ROW_GROUP - 1) // ROW_GROUP
    use_pool    = batch_bytes < POOL_THRESH

    # Create resources BEFORE timing
    if use_pool:
        pool = dgen_py.BufferPool()                   # NOT timed
        gen  = None
    else:
        pool = None
        gen  = dgen_py.Generator(size=file_bytes)     # NOT timed

    t0   = time.perf_counter()
    sink = pa.BufferOutputStream()

    with pq.ParquetWriter(sink, schema, compression=None) as w:
        for bi in range(num_batches):
            cur_rows  = min(ROW_GROUP, num_samples - bi * ROW_GROUP)
            cur_bytes = cur_rows * elem_size

            if pool is not None:
                raw = pool.next_slice(cur_bytes)
                arr = np.frombuffer(raw, dtype=np.uint8)
            else:
                gen.fill_chunk(buf)
                arr = np.frombuffer(buf, dtype=np.uint8)[:cur_bytes]

            arrow_flat = pa.array(arr, type=pa.uint8())
            arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
            w.write_table(pa.table({"data": arrow_data}), row_group_size=ROW_GROUP)

    encoded = sink.getvalue()
    return time.perf_counter() - t0, len(encoded)

def bench_parquet_mem(label, num_samples):
    file_bytes  = num_samples * ELEM_SIZE
    batch_bytes = min(ROW_GROUP, num_samples) * ELEM_SIZE
    ram_after   = _human(batch_bytes if batch_bytes < POOL_THRESH else CHUNK_SIZE)
    buf         = dgen_py.create_bytearrays(count=1, size=CHUNK_SIZE)[0]

    print(f"\n  [{label}  {_human(file_bytes)}]")

    # BEFORE
    _to_mem_before_timed(num_samples, ELEM_SIZE)                       # warm-up
    results_b = [_to_mem_before_timed(num_samples, ELEM_SIZE) for _ in range(REPS)]
    times_b   = [r[0] for r in results_b]
    enc_b     = results_b[0][1]
    avg_b     = _avg(times_b)
    print(f"    BEFORE  mem: {' '.join(f'{t:.3f}s' for t in times_b)}"
          f"  -> avg {_tput(file_bytes, avg_b):>12}  encoded {_human(enc_b)}"
          f"  peak RAM ~{_human(file_bytes)}")

    # AFTER
    _to_mem_after_timed(num_samples, ELEM_SIZE, buf)                   # warm-up
    results_a = [_to_mem_after_timed(num_samples, ELEM_SIZE, buf) for _ in range(REPS)]
    times_a   = [r[0] for r in results_a]
    enc_a     = results_a[0][1]
    avg_a     = _avg(times_a)
    print(f"    AFTER   mem: {' '.join(f'{t:.3f}s' for t in times_a)}"
          f"  -> avg {_tput(file_bytes, avg_a):>12}  encoded {_human(enc_a)}"
          f"  peak RAM ~{ram_after}")

    print(f"    Speedup: {avg_b/avg_a:.1f}x")


# ── SECTION 3: Full end-to-end to NVMe ───────────────────────────────────────

def _to_disk_before_timed(num_samples, elem_size, path):
    file_bytes  = num_samples * elem_size
    schema      = _schema(elem_size)
    num_batches = (num_samples + ROW_GROUP - 1) // ROW_GROUP

    gen = dgen_py.Generator(size=file_bytes)          # NOT timed

    t0   = time.perf_counter()
    bv   = gen.get_chunk(file_bytes)
    flat = np.frombuffer(bv, dtype=np.uint8).copy()
    del bv, gen

    arrow_flat = pa.array(flat, type=pa.uint8())
    arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
    full_table = pa.table({"data": arrow_data})
    del flat

    with pq.ParquetWriter(path, schema, compression=None) as w:
        for bi in range(num_batches):
            start    = bi * ROW_GROUP
            cur_rows = min(ROW_GROUP, num_samples - start)
            w.write_table(full_table.slice(start, cur_rows), row_group_size=ROW_GROUP)

    return time.perf_counter() - t0

def _to_disk_after_timed(num_samples, elem_size, path, buf):
    file_bytes  = num_samples * elem_size
    batch_bytes = min(ROW_GROUP, num_samples) * elem_size
    schema      = _schema(elem_size)
    num_batches = (num_samples + ROW_GROUP - 1) // ROW_GROUP
    use_pool    = batch_bytes < POOL_THRESH

    # Create resources BEFORE timing
    if use_pool:
        pool = dgen_py.BufferPool()                   # NOT timed
        gen  = None
    else:
        pool = None
        gen  = dgen_py.Generator(size=file_bytes)     # NOT timed

    t0 = time.perf_counter()

    with pq.ParquetWriter(path, schema, compression=None) as w:
        for bi in range(num_batches):
            cur_rows  = min(ROW_GROUP, num_samples - bi * ROW_GROUP)
            cur_bytes = cur_rows * elem_size

            if pool is not None:
                raw = pool.next_slice(cur_bytes)
                arr = np.frombuffer(raw, dtype=np.uint8)
            else:
                gen.fill_chunk(buf)
                arr = np.frombuffer(buf, dtype=np.uint8)[:cur_bytes]

            arrow_flat = pa.array(arr, type=pa.uint8())
            arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
            w.write_table(pa.table({"data": arrow_data}), row_group_size=ROW_GROUP)

    return time.perf_counter() - t0

def bench_parquet_disk(label, num_samples, tmpdir):
    file_bytes  = num_samples * ELEM_SIZE
    batch_bytes = min(ROW_GROUP, num_samples) * ELEM_SIZE
    pb  = os.path.join(tmpdir, f"{label.replace(' ','_')}_b.parquet")
    pa_ = os.path.join(tmpdir, f"{label.replace(' ','_')}_a.parquet")
    ram_after = _human(batch_bytes if batch_bytes < POOL_THRESH else CHUNK_SIZE)
    buf = dgen_py.create_bytearrays(count=1, size=CHUNK_SIZE)[0]

    print(f"\n  [{label}  {_human(file_bytes)}]")

    # BEFORE
    _to_disk_before_timed(num_samples, ELEM_SIZE, pb); os.unlink(pb)
    times_b = []
    for _ in range(REPS):
        times_b.append(_to_disk_before_timed(num_samples, ELEM_SIZE, pb))
        os.unlink(pb)
    avg_b = _avg(times_b)
    print(f"    BEFORE  nvme: {' '.join(f'{t:.3f}s' for t in times_b)}"
          f"  -> avg {_tput(file_bytes, avg_b):>12}  peak RAM ~{_human(file_bytes)}")

    # AFTER
    _to_disk_after_timed(num_samples, ELEM_SIZE, pa_, buf); os.unlink(pa_)
    times_a = []
    for _ in range(REPS):
        times_a.append(_to_disk_after_timed(num_samples, ELEM_SIZE, pa_, buf))
        os.unlink(pa_)
    avg_a = _avg(times_a)
    print(f"    AFTER   nvme: {' '.join(f'{t:.3f}s' for t in times_a)}"
          f"  -> avg {_tput(file_bytes, avg_a):>12}  peak RAM ~{ram_after}")

    print(f"    Speedup: {avg_b/avg_a:.1f}x  RAM savings: {max(1, file_bytes // max(batch_bytes, CHUNK_SIZE))}x")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_benchmarks(gen_only=False, include_10gb=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="dlio_bench_", dir=OUTPUT_DIR)
    sep = "=" * 80

    try:
        ncpus = os.cpu_count() or 1
        print(f"\n{sep}")
        print(f"  dgen-py v{dgen_py.__version__}  |  {ncpus} logical CPUs")
        print(f"  elem_size: {_human(ELEM_SIZE)}/sample  "
              f"|  chunk_size: {_human(CHUNK_SIZE)}  "
              f"|  pool_thresh: {_human(POOL_THRESH)}  "
              f"|  reps: {REPS}+warmup")
        print(f"  Generator/Pool creation is NOT timed; only fill calls are timed.")
        print(sep)

        active = [(lbl, n) for lbl, n, ok in CASES if ok or include_10gb]

        print("\n== SECTION 1: Pure data generation (no Arrow, no disk) ==")
        print("   BEFORE: Generator(size=N) -> get_chunk(N)")
        print("   AFTER  (<32 MB): BufferPool() -> next_slice(N)")
        print("   AFTER  (>=32 MB): Generator(size=N) -> fill_chunk(64 MB) loop")
        for lbl, n in active:
            bench_gen_only(lbl, n)

        if not gen_only:
            print("\n== SECTION 2: Parquet encode to memory (gen + Arrow + encode, no disk) ==")
            for lbl, n in active:
                bench_parquet_mem(lbl, n)

            print(f"\n== SECTION 3: Full end-to-end to NVMe ({OUTPUT_DIR}) ==")
            print(f"   Note: vSAN ceiling on this VM is ~2 GB/s.")
            for lbl, n in active:
                bench_parquet_disk(lbl, n, tmpdir)

        print(f"\n{sep}\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── pytest ────────────────────────────────────────────────────────────────────

_LARGE = os.environ.get("DLIO_BENCH_LARGE", "0") == "1"
_LARGE_REASON = "Set DLIO_BENCH_LARGE=1"

def test_parquet_bench_1mb():
    bench_gen_only("1 MB", 2)
    bench_parquet_mem("1 MB", 2)

def test_parquet_bench_10mb():
    bench_gen_only("10 MB", 20)
    bench_parquet_mem("10 MB", 20)

def test_parquet_bench_100mb():
    bench_gen_only("100 MB", 200)
    bench_parquet_mem("100 MB", 200)

@pytest.mark.skipif(not _LARGE, reason=_LARGE_REASON)
def test_parquet_bench_1gb():
    bench_gen_only("1 GB", 2048)
    bench_parquet_mem("1 GB", 2048)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-only",     action="store_true",
                   help="Section 1 only: pure data generation, no Parquet or disk")
    p.add_argument("--include-10gb", action="store_true",
                   help="Include the 10 GB case")
    args = p.parse_args()
    run_benchmarks(gen_only=args.gen_only, include_10gb=args.include_10gb)

if __name__ == "__main__":
    main()
