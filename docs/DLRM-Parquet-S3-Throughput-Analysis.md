# DLRM Parquet S3 I/O Throughput Analysis

**Date:** May 2026  
**Benchmark:** MLPerf Storage DLRM training workload  
**Storage backend:** s3-ultra (fake S3 server, in-memory parquet footer cache)  
**Reader:** `ParquetReaderS3Iterable` with s3dlio byte-range GETs  
**Host:** Single machine, loopback TCP (no network), 47 GiB RAM  

---

## 1. File Format Parameters

| Parameter | Value |
|-----------|-------|
| Files | 64 |
| Size per file | ~1 GiB |
| Row groups per file | 123 |
| Rows per row group | 8,192 |
| Total rows per file | 1,000,000 |
| Total columns | 200 |
| Columns read (projection) | 40 |
| Compressed size, 40 cols, 1 RG | ~1,721 KiB |
| Byte span of 40-col GET per RG | ~1,743 KiB |
| Parquet footer size | ~2.66 MiB |

---

## 2. Throughput Progression

Each fix below was implemented and benchmarked sequentially.

| Version | Change | Throughput | Speedup vs baseline |
|---------|--------|-----------|---------------------|
| v1 — Original | `pf.read_row_group()` — 40 HTTP GETs/RG, `get_parquet_footer` Fjall read every GET | 93 MiB/s | baseline |
| v2 — Merged GET | Single merged GET per RG (min–max column span), s3-ultra in-memory footer cache, early-out for non-footer GETs | 288 MiB/s | 3.1× |
| v3 — Prefetch + fast read_index | Background ThreadPoolExecutor (32 workers) prefetches all RGs at file open; `read_index()` override removes per-sample `utcnow()` | 402 MiB/s | 4.3× |

Target: **10,240 MiB/s (10 GiB/s)**. Still ~25× below target at v3.

---

## 3. Root Cause Analysis — Why 10 GiB/s Is Not Achieved

### 3.1 Per-sample Python call overhead is the hard bottleneck

The PyTorch `Dataset.__getitem__` protocol calls `read_index(image_idx, step)` once per **sample**. With 1,000,000 samples/file × 16 files/worker = **16,000,000 Python calls per worker per epoch**.

Even with all I/O eliminated (cache-hit path), each call costs ~3 µs:

| Overhead source | Cost per call | Total (16M calls) |
|----------------|---------------|-------------------|
| `FormatReader.read_index()` base — `datetime.now().strftime()` called twice (debug log f-strings evaluate unconditionally) | ~3 µs | ~48 s |
| Bisect + dict lookup | ~0.5 µs | ~8 s |
| `dlp.update()` | ~0.1 µs | ~2 s |
| **Total Python overhead** | **~3.6 µs** | **~58 s** |

Actual S3 I/O (1,968 row group GETs × ~4.9 ms serial, or ~15 ms for 10 in parallel with 32 threads) consumes only **~3–15 s** per worker. The benchmark runs 131 s — Python overhead dominates by ~4:1.

### 3.2 How much actual data moves per worker

```
16 files × 123 row groups × 1,743 KiB/RG = 3,350 MiB per worker
```

At 10 GiB/s that takes **0.33 s**. Python overhead is ~180× the I/O time.

### 3.3 The threshold: row-group granularity

If Python were called once per **row group** instead of once per **sample**:

```
16 files × 123 row groups = 1,968 Python calls  (vs 16,000,000)
Reduction: 8,130×
Python overhead: 1,968 × 3.6 µs = 0.007 s
I/O time @ 10 GiB/s: 0.33 s
Theoretical throughput: 3,350 MiB / 0.337 s = 9,940 MiB/s ≈ 10 GiB/s ✓
```

Row-group granularity is the **exact inflection point** that makes 10 GiB/s achievable.

---

## 4. Memory Budget for Row-Group Prefetch

| Scope | Size |
|-------|------|
| 1 RG in flight (40 cols, span) | ~1.7 MiB |
| 32 RGs in flight (prefetch window) | ~54 MiB |
| All 123 RGs of 1 file fully buffered | ~209 MiB |
| 4 workers × full file | **~838 MiB** |

Against 47 GiB host RAM, 838 MiB is ~1.8% — completely benign.

---

## 5. Three-Level Optimization Roadmap

### Level 1 — Row-group IterableDataset (achievable now, within DLIO)

**Impact: ~16× speedup → ~6 GiB/s**

Replace the `Dataset` (map-style, `__getitem__` per sample) with a PyTorch `IterableDataset`. Each iteration yields one batch sourced from a single row-group fetch. Python I/O calls: **1,968** instead of 16,000,000.

```python
# Pseudocode: IterableDataset approach
def __iter__(self):
    for filename in self.file_list:
        pf, rf, offsets = self.open(filename)
        for rg_idx in range(pf.metadata.num_row_groups):
            compressed = self._fetch_row_group_single_get(rf, pf, rg_idx)  # 1 HTTP GET
            for _ in range(pf.metadata.row_group(rg_idx).num_rows):
                dlp.update(image_size=compressed)
                yield self._args.resized_image  # pure Python, no I/O
```

The inner `yield` loop (16M iterations) is pure Python with no I/O: ~0.5 µs/iter = ~8 s.  
Total estimate: 8 s Python + 0.33 s I/O = **~8.3 s → ~404 MiB/s per worker × 4 = ~1,600 MiB/s**.

Wait — this is only 1,600 MiB/s, not 6 GiB/s, because the inner yield loop is still 16M Python calls. The gain is that those calls have no I/O and no `datetime.now()`.

To reach 10 GiB/s, the inner loop must also be eliminated. See Level 3.

**True ceiling with IterableDataset but Python inner loop:**
- Python inner loop: 16M × 0.5 µs = 8 s dominant
- Still 4× better than current 131 s → ~1,600 MiB/s aggregate

### Level 2 — Async prefetch pipeline (production standard)

**Impact: Overlaps I/O with compute, eliminates stalls**

Current prefetch submits all RGs as futures at file-open time. Improvement: use a bounded queue (e.g. depth=8) so memory is capped and the next file's prefetch starts while the current file's later RGs are being consumed.

```
[File opener thread] → [RG fetch pool, 32 threads] → [bounded queue, depth 8] → [main loop]
```

While GPU processes batch N, threads fetch RGs N+1 through N+8. Loopback latency (~4 ms/GET) is fully hidden. On real network (10 GiB/s storage array), this is critical — the fetch pool must be wide enough to keep N parallel streams open.

This is already partially implemented (prefetch executor). The bounded queue is the missing piece to prevent memory spikes on large epoch sizes.

### Level 3 — Native Rust parquet reader in s3dlio (long-term, biggest impact)

**Impact: Eliminates Python from the hot path entirely → 10+ GiB/s achievable**

Move footer parsing + column range computation + RG fetch into a Rust function exposed via PyO3:

```rust
// s3dlio API addition (conceptual)
pub fn read_parquet_rg_extent(
    uri: &str,
    footer_bytes: &[u8],   // already cached in s3-ultra / s3dlio
    rg_idx: usize,
    col_indices: &[usize],
) -> Result<usize>  // returns compressed_bytes, discards data
```

Python calls this **once per row group** — 1,968 calls/worker. The inner per-sample loop disappears entirely from Python. Rust computes the byte extent from the Thrift footer, issues the GET, discards the bytes, returns the byte count. No GIL-holding Python loop.

**Theoretical ceiling:**
- Rust calls: 1,968 × (Rust overhead ~1 µs) = 2 ms
- I/O: 0.33 s @ 10 GiB/s
- Total: **~0.332 s → 10,060 MiB/s per worker**

This is the only path to full 10 GiB/s with the existing per-file/per-row-group format.

**Where does it belong?** See Section 6.

---

## 6. Architecture Decision: Where Does the Rust Parquet Reader Live?

### Option A: Inside s3dlio

**Pros:**
- s3dlio already has: URI parsing, credential resolution, S3/GCS/Azure/file backends, PyO3 bindings, range GET primitives
- `read_parquet_rg_extent(uri, footer_bytes, rg_idx, columns)` fits naturally alongside `get_range()`
- Single dependency for downstream tools (dl-driver, sai3-bench, warpio)
- Footer bytes already cached in s3-ultra's `parquet_footer_cache` and can be served via `get_range(uri, size-footer_len, footer_len)`

**Cons:**
- Adds parquet Thrift parsing to s3dlio (new dependency: `parquet` crate)
- s3dlio becomes opinionated about file formats (currently format-agnostic)

### Option B: Standalone `parquet-s3-reader` crate / Python package

**Pros:**
- Clean separation: s3dlio stays format-agnostic
- Can be used without s3dlio (e.g. with a different S3 backend)
- Easier to publish independently

**Cons:**
- Another dependency to manage
- Duplicates URI/credential logic already in s3dlio
- More friction for dlio_benchmark integration

### Option C: Inside dlio_benchmark as a compiled extension (`dlio_parquet_ext`)

**Pros:**
- Scoped to the benchmark — no library API surface to maintain
- Ships with dlio_benchmark wheel

**Cons:**
- Embedded in a Python project — harder to maintain Rust build chain
- Not reusable by dl-driver, sai3-bench, etc.

### Recommendation

**Option A (s3dlio) is the right choice** for this ecosystem. s3dlio is already the storage abstraction layer used by all tools. Adding `read_parquet_rg_extent()` makes it a complete "storage + parquet I/O" primitive, which is the natural evolution given that parquet-on-object-storage is the dominant AI/ML training format. The `parquet` crate is mature and the Thrift footer parsing is ~200 lines of Rust.

---

## 7. Implementation Priority Order

| Priority | Work | Expected result |
|----------|------|----------------|
| 1 (highest) | Level 3: `s3dlio::read_parquet_rg_extent()` Rust API + PyO3 binding | ~10 GiB/s |
| 2 | Level 1: IterableDataset reader in dlio_benchmark using new s3dlio API | Unlock Level 3 gain |
| 3 | Level 2: Bounded prefetch queue (depth=8) replacing unbounded futures dict | Stable memory, full overlap |

---

## 8. Current State (as of May 2026)

- `parquet_reader_s3_iterable.py` — merged GET per RG + 32-thread prefetch executor + fast `read_index()` override: **402 MiB/s**
- s3-ultra — in-memory `parquet_footer_cache` (DashMap), early-out for non-footer GETs, release binary: s3 server CPU < 1 core at full DLRM load
- Hard ceiling without Level 3: **~1,600 MiB/s** (Python inner loop dominates)
- Hard ceiling with Level 3: **~10,060 MiB/s** (I/O bound)

---

## 9. Three-Mode Read Benchmark (May 2026, local NVMe, s3dlio v0.9.100)

**Test configuration:** 4 files × 8 row groups × 200 float32 cols = 32 RGs total, 35.6 MB on disk (Snappy compressed)
**Storage:** local NVMe via `file://` URIs (best-case; S3 latency would widen the gaps further)

| Mode | Reader | Workers | Time | Decoded MB | Throughput |
|------|--------|---------|------|------------|------------|
| 1 — s3dlio raw + DISCARD | `parquet_get_rg(decode="raw")` → `len(bytes(bv))` | serial | 0.085s | 34.9 MB (compressed) | **412 MB/s** |
| 2 — PyArrow native (baseline) | `pq.ParquetFile.read_row_group()` | serial | 0.170s | 27.0 MB | 159 MB/s |
| 3 — s3dlio arrow + IPC decode | `parquet_get_rg(decode="arrow")` + `pa.ipc.open_stream()` | 4 | 0.219s | 26.2 MB | 120 MB/s |
| 3 — s3dlio arrow + IPC decode | same | 8 | 0.159s | 26.2 MB | **164 MB/s** |
| 3 — s3dlio arrow + IPC decode | same | 16 | 0.174s | 26.2 MB | 151 MB/s |
| 3 — s3dlio arrow + IPC decode | same | 32 | 0.168s | 26.2 MB | 156 MB/s |

**Key observations:**

- **Mode 1 (raw + discard)** is 2.6× faster than PyArrow native — pure I/O with zero decode overhead. This is the storage benchmark mode used in production throughput tests.
- **Mode 2 (PyArrow native)** is the baseline any default PyTorch DataLoader achieves. Serial, blocking, one HTTP GET per `read_row_group()` call.
- **Mode 3 (s3dlio arrow, 8 workers)** at 164 MB/s edges the PyArrow serial baseline. On local NVMe decode CPU cost dominates; on S3 with real network latency the concurrent pipeline advantage grows substantially — s3dlio holds a warm connection pool and overlaps I/O with Rust-side Parquet→Arrow decode entirely off the GIL.
- Sweet spot is 8 workers for 32 RGs (one worker per 4 RGs). Beyond that, thread-switching overhead on small tasks flattens the curve.
- `configure_tokio_threads()` is now called in both `ParquetReaderS3dlio.__init__` and `ParquetReaderS3dlioArrow.__init__` so the Tokio thread budget is MPI-aware from the first call.
