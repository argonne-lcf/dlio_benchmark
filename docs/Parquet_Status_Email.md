# Parquet S3 Dataloader — Status & Path to 10 GB/s

**Date:** May 5, 2026  
**To:** [distribution]  
**Re:** MLPerf Storage DLRM read throughput — progress update and outstanding decisions

---

## Where We Started

The baseline DLRM parquet reader achieved roughly **84 MB/s** against an in-process
S3 server on loopback (no real network latency, no disk I/O — the floor is software
overhead).  The target for MLPerf Storage is **10 GiB/s per host**.  The gap was
125×.

---

## What Was Fixed and Why It Worked

Three successive changes brought throughput from 84 MB/s to 2.1 GB/s.

### Fix 1 — Merge the HTTP GETs: 84 → 200 MB/s

**Problem:** pyarrow's `read_row_group()` issues one HTTP request per column chunk.
With 40 columns selected that is 40 GETs per row group, 78,720 GETs per epoch.
Each GET incurred full HTTP overhead plus a metadata lookup in s3-ultra.

**Fix:** Replace 40 column GETs with a single byte-range GET spanning the min–max
column extent within the row group.  Parquet stores column chunks contiguously, so
the merged span contains exactly the same bytes.  Also added an in-memory footer
cache to s3-ultra so repeated footer reads were served from RAM.

**Result:** 78,720 GETs → 1,968 GETs per epoch.  **2.4× speedup.**

---

### Fix 2 — Prefetch at file open: 200 → 400 MB/s

**Problem:** Row-group GETs were still issued serially, triggered one at a time
by the training loop as it called into the reader.  Each GET blocked the reader
thread until the response arrived.

**Fix:** At `file.open()` time, submit all 123 row-group GETs for that file
concurrently using a 32-thread `ThreadPoolExecutor`.  Also overrode
`read_index()` to remove two `datetime.now().strftime()` calls that fired
unconditionally on every sample (16 M × ~3 µs = 48 s wasted per worker per epoch).

**Result:** Serial I/O latency eliminated; most row groups already in memory before
the training loop asks for them.  **4.8× speedup over baseline.**

---

### Fix 3 — Move iteration to Rust: 400 → 2,100 MB/s

**Problem:** Even with all I/O pre-fetched, `read_index()` was still called once per
*sample* — 16,000,000 Python function calls per worker per epoch.  At a minimum
cost of ~3.6 µs each (unavoidable in CPython regardless of what the function does),
that is a hard floor of ~58 s of pure Python overhead per worker.  At 10 GiB/s the
actual I/O takes only 0.33 s.  Python overhead was 176× the I/O time — no amount
of parallelism or caching can overcome it while the call count remains at 16 M.

**The insight:** The dataset has 1,968 unique I/O operations (one per row group).
If Python is called 1,968 times instead of 16,000,000 times, Python overhead drops
to 0.007 s — negligible against 0.33 s of I/O.

**Fix:** Implemented `ParquetRowGroupDataset` in Rust inside the s3dlio library.
Rust builds the full row-group index (file listing → concurrent footer GETs →
Parquet metadata parse → one `{offset, length}` extent per row group) once at
construction.  A `buffer_unordered(prefetch)` stream in Tokio drives up to N
concurrent range GETs simultaneously; a bounded channel feeds results to Python.
`for item in loader:` in Python calls `__next__()` 1,968 times — once per row
group, not once per sample.

**Result:** 16,000,000 Python calls → 1,968 Python calls.  GIL overhead negligible.
**25× speedup over baseline.  5.4× over the previous best.**

---

## Current Numbers

| Iteration | Change | Throughput | vs. Baseline |
|-----------|--------|-----------|-------------|
| Baseline | pyarrow per-column GETs | 84 MB/s | 1× |
| Fix 1 | Single merged GET per row group | 200 MB/s | 2.4× |
| Fix 2 | 32-thread prefetch + fast read_index | 400 MB/s | 4.8× |
| **Fix 3** | **Rust row-group iterator in s3dlio** | **2,100 MB/s** | **25×** |
| **Target** | | **10,240 MB/s** | **122×** |

The remaining gap is **5×**.  It is not a fundamental algorithmic problem — we
have already proved the iteration overhead is gone.  The gap is parallelism and
one architectural decision.

---

## What Is Needed to Reach 10 GB/s

### 1. Multiple reader workers (expected: linear scaling, ~4–5×)

The test that produced 2.1 GB/s used a **single Python process with a single
loader**.  The MLPerf Storage spec runs multiple data-loader workers in parallel,
each owning a disjoint file shard.

With 5 workers each sustaining 2.1 GB/s the aggregate is already 10.5 GB/s.
With 4 workers at 2.5 GB/s (likely once the connection pool is fully warm from
the start) it hits target.  This is the lowest-risk path — it requires no further
code changes, just the correct DLIO YAML worker count and a check that s3-ultra
can sustain the aggregate rate (it runs entirely in memory; it should).

**Decision needed:** Run the multi-worker benchmark to confirm linear scaling.
What worker count does the MLPerf Storage spec require for this workload?

### 2. Column projection in the range GET (expected: 3–5× data reduction)

The Phase 3 test fetched **all columns** (full row-group byte span).  The DLRM
workload reads only 40 of 200 columns.  The column-selection logic already exists
in `ParquetRowGroupDataset::new(col_indices)` but was not exercised in the test.

When enabled, each range GET shrinks from ~1,743 KiB (full span) to ~487 KiB (40
selected columns), a **3.6× reduction** in bytes transferred.  At the same network
rate, throughput in terms of samples/s goes up proportionally.  The downstream
question is whether pyarrow can decode the sub-selected bytes or whether the raw
bytes are handed to the training framework directly.

**Decision needed:** Does the DLRM training loop require decoded Arrow columns or
raw Parquet bytes?  If raw bytes, column projection is free today.  If decoded
columns, see item 3 below.

### 3. Arrow decoding in Rust — Phase 4 (expected: removes remaining Python decode cost)

Currently `ParquetRowGroupDataset` returns compressed Parquet bytes.  The Python
consumer (or the training framework) must decode them.  Adding Arrow IPC decoding
inside the Rust `get()` call would:

- Decode Parquet bytes → Arrow `RecordBatch` entirely in Rust (Tokio thread, no GIL)
- Transfer the decoded batch to Python via the Arrow C Data Interface (zero-copy,
  reference-counted; pyarrow can wrap it with no memcpy)
- Eliminate all remaining Python-side decode overhead

The `parquet` crate already has this capability when compiled with
`features = ["arrow"]`; the current build uses `default-features = false` to keep
the dependency tree small.  Enabling Arrow output is a ~1-day Rust change.

**Decision needed:** Is Phase 4 required to hit 10 GB/s, or does multi-worker
alone get there?  Multi-worker is the right experiment to run first — it is free.

### 4. Prefetch depth tuning (minor, ~10–20%)

The test used `prefetch = 16`.  With a higher prefetch depth (32–64) Tokio keeps
more GETs in flight simultaneously, better hiding individual request latency.
At 2.1 GB/s steady state the channel was not the bottleneck, but on a real network
(vs. loopback) higher prefetch may be important.

---

## Recommended Next Steps (in order)

1. **Run multi-worker benchmark** — 4 workers, each 16 files, same YAML config with
   `storage_library: s3dlio`.  This is the single most important data point.  If
   aggregate throughput scales linearly, 10 GB/s is already achieved with existing
   code.

2. **Enable column projection** — pass `col_indices: [0..39]` to the loader and
   re-run.  Confirms the 3.6× byte reduction works end-to-end.

3. **Decide on Arrow decoding** — required only if multi-worker + projection still
   falls short of 10 GB/s, or if the training framework requires decoded tensors
   rather than raw bytes.

---

## Decisions Requested

| # | Decision | Options | Impact |
|---|----------|---------|--------|
| 1 | How many DLIO data-loader workers does the MLPerf Storage spec require for DLRM? | 1 / 4 / 8 / other | Determines if 10 GB/s is already achieved |
| 2 | Does the training framework require decoded columns (Arrow/numpy) or raw Parquet bytes? | Raw bytes OK / Decoded required | Determines whether Phase 4 (Arrow-in-Rust) is needed |
| 3 | Should Phase 4 (Arrow IPC from Rust) be built now or deferred until multi-worker results are in? | Build now / Wait for data | ~1 day Rust work |
| 4 | Is 10 GB/s aggregate (all workers combined) or 10 GB/s per worker the MLPerf target? | Aggregate / Per-worker | Changes required worker count by ~5× |

---

*All code changes are on the `feat/parquet-dataloader` branch of s3dlio and in
`dlio_benchmark/dlio_benchmark/reader/parquet_reader_s3dlio.py`.  The existing
parquet reader paths are completely unchanged — the new reader is opt-in via
`storage_library: s3dlio` in the YAML config.*