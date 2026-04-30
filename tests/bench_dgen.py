#!/usr/bin/env python3
"""bench_dgen.py — dgen-py data generation throughput: BEFORE vs. AFTER.

Shows the raw generation speed improvement from the new dgen-py 0.2.3 API:

  BEFORE: Generator(size=N).get_chunk(N)
    Allocates the full file in RAM, generates everything, returns a buffer.
    A new Rayon thread pool is created on every call.

  AFTER (file < 32 MB): BufferPool().next_slice(N)
    Zero-copy slice from a rolling buffer.  No thread pool overhead.

  AFTER (file >= 32 MB): Generator(size=N) + fill_chunk(64 MB loop)
    64 MB chunks, Rayon thread pool reused across all chunks.
    Peak RAM is always 64 MB regardless of file size.

Timing discipline:
  Generator/Pool creation is NOT included in the timed window.
  Only the fill call(s) are timed, so we measure pure generation speed.
  A fresh Generator is created per rep (current_pos is not reset by
  set_seed(), so a drained Generator cannot be reused).

Usage
-----
  python tests/bench_dgen.py                  # all sizes up to 1 GB
  python tests/bench_dgen.py --include-10gb   # add 10 GB case
  python -m pytest tests/bench_dgen.py -s -v
"""

import argparse
import os
import time

import pytest

import dgen_py

# ── Constants ─────────────────────────────────────────────────────────────────
ELEM_SIZE   = 512 * 1024        # 512 KB per sample (representative DL training data)
CHUNK_SIZE  = 64 * 1024 * 1024  # fill_chunk buffer (>= 32 MB required by dgen-py)
POOL_THRESH = 32 * 1024 * 1024  # use BufferPool strictly below this threshold
REPS        = 3                 # timed reps per case (+ 1 warm-up)

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

def _avg(lst):
    return sum(lst) / len(lst)

# ── Timed generation primitives ───────────────────────────────────────────────

def _before(file_bytes):
    """BEFORE: create Generator (not timed) -> time get_chunk only."""
    gen = dgen_py.Generator(size=file_bytes)          # NOT timed
    t0  = time.perf_counter()
    bv  = gen.get_chunk(file_bytes)
    t   = time.perf_counter() - t0
    del bv, gen
    return t

def _after_pool(file_bytes):
    """AFTER small: create BufferPool (not timed) -> time next_slice only."""
    pool = dgen_py.BufferPool()                       # NOT timed
    t0   = time.perf_counter()
    raw  = pool.next_slice(file_bytes)
    t    = time.perf_counter() - t0
    del raw, pool
    return t

def _after_stream(file_bytes, buf):
    """AFTER large: create Generator (not timed) -> time fill_chunk loop."""
    gen = dgen_py.Generator(size=file_bytes)          # NOT timed
    t0  = time.perf_counter()
    while not gen.is_complete():
        gen.fill_chunk(buf)
    t = time.perf_counter() - t0
    del gen
    return t

# ── Per-case benchmark ────────────────────────────────────────────────────────

def bench_case(label, num_samples):
    file_bytes = num_samples * ELEM_SIZE
    use_pool   = file_bytes < POOL_THRESH
    buf        = None if use_pool else dgen_py.create_bytearrays(count=1, size=CHUNK_SIZE)[0]

    print(f"\n  [{label}  {_human(file_bytes)}]")

    # BEFORE
    _before(file_bytes)                                # warm-up
    times_b = [_before(file_bytes) for _ in range(REPS)]
    avg_b   = _avg(times_b)
    print(f"    BEFORE  get_chunk:        "
          f"{' '.join(f'{t:.3f}s' for t in times_b)}"
          f"  avg {_tput(file_bytes, avg_b):>12}   peak RAM ~{_human(file_bytes)}")

    # AFTER
    if use_pool:
        after_label = "pool.next_slice:  "
        ram_after   = _human(file_bytes)
        _after_pool(file_bytes)                        # warm-up
        times_a = [_after_pool(file_bytes) for _ in range(REPS)]
    else:
        after_label = f"fill_chunk({_human(CHUNK_SIZE)}):"
        ram_after   = _human(CHUNK_SIZE)
        _after_stream(file_bytes, buf)                 # warm-up
        times_a = [_after_stream(file_bytes, buf) for _ in range(REPS)]
    avg_a = _avg(times_a)
    print(f"    AFTER   {after_label} "
          f"{' '.join(f'{t:.3f}s' for t in times_a)}"
          f"  avg {_tput(file_bytes, avg_a):>12}   peak RAM ~{ram_after}")

    speedup  = avg_b / avg_a
    ram_save = max(1, file_bytes // max(file_bytes if use_pool else CHUNK_SIZE, 1))
    print(f"    Speedup: {speedup:.1f}x   RAM savings: {max(1, file_bytes // (CHUNK_SIZE if not use_pool else file_bytes))}x")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_benchmarks(include_10gb=False):
    sep = "=" * 72
    ncpus = os.cpu_count() or 1
    print(f"\n{sep}")
    print(f"  dgen-py v{dgen_py.__version__}  |  {ncpus} logical CPUs")
    print(f"  elem_size: {_human(ELEM_SIZE)}/sample"
          f"  |  chunk: {_human(CHUNK_SIZE)}"
          f"  |  pool_thresh: {_human(POOL_THRESH)}"
          f"  |  reps: {REPS}+warmup")
    print(f"  Timing: Generator/Pool creation excluded; only fill call(s) timed.")
    print(sep)

    print("\n  Method comparison:")
    print("    BEFORE : Generator(size=N).get_chunk(N)          full-file RAM, new thread pool each call")
    print("    AFTER  : BufferPool().next_slice(N)   [<32 MB]   zero-copy slice, no thread pool")
    print("    AFTER  : Generator(size=N)+fill_chunk [>=32 MB]  64 MB peak RAM, Rayon pool reused\n")

    active = [(lbl, n) for lbl, n, ok in CASES if ok or include_10gb]
    for lbl, n in active:
        bench_case(lbl, n)

    print(f"\n{sep}\n")


# ── pytest ────────────────────────────────────────────────────────────────────

def test_bench_1mb():
    bench_case("1 MB", 2)

def test_bench_10mb():
    bench_case("10 MB", 20)

def test_bench_100mb():
    bench_case("100 MB", 200)

_LARGE = os.environ.get("DLIO_BENCH_LARGE", "0") == "1"

@pytest.mark.skipif(not _LARGE, reason="Set DLIO_BENCH_LARGE=1")
def test_bench_1gb():
    bench_case("1 GB", 2048)

@pytest.mark.skipif(not _LARGE, reason="Set DLIO_BENCH_LARGE=1")
def test_bench_10gb():
    bench_case("10 GB", 20480)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-10gb", action="store_true",
                   help="Include the 10 GB case")
    args = p.parse_args()
    run_benchmarks(include_10gb=args.include_10gb)

if __name__ == "__main__":
    main()
