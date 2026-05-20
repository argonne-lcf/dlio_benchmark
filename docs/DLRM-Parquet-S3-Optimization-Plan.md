# DLRM Parquet S3 Optimization Plan

**Date:** May 5, 2026  
**Status:** Planning — awaiting implementation start  
**Target:** 10 GiB/s aggregate throughput (from 402 MiB/s baseline)

---

## Problem Recap

| Stage | Throughput | Bottleneck |
|-------|-----------|-----------|
| Baseline | 93 MiB/s | Fjall LSM read per GET + 40 HTTP GETs/RG |
| After s3-ultra + merged GET | 288 MiB/s | 16M Python `read_index()` calls × utcnow() overhead |
| After prefetch + fast `read_index()` | 402 MiB/s | 16M Python calls × 0.5 µs = 8s hard Python floor |
| **Target** | **~10,000 MiB/s** | I/O bound at ~0.33s (1,968 RG fetches, ~1.7 MiB each) |

The remaining 25× gap is **pure Python overhead** — `read_index()` is called once per *sample* (16 million times per worker), but only 1,968 actual I/O operations are needed. Closing this gap requires reducing Python call count from 16M → ~1,968.

---

## Key Architectural Insight: s3dlio DataLoader Extension

s3dlio already has a production-quality `DataLoader` framework in `src/data_loader/`:

```
src/data_loader/
  dataset.rs             ← Dataset trait (get, len, as_stream, keys)
  dataloader.rs          ← DataLoader<D: Dataset> generic wrapper
  async_pool_dataloader.rs ← AsyncPoolDataLoader + UnifiedDataLoader
  options.rs             ← LoaderOptions (prefetch, max_inflight_parts, num_workers, part_size, …)
  s3_bytes.rs            ← S3BytesDataset implements Dataset<Item=Bytes>
  fs_bytes.rs            ← FileSystemBytesDataset
  directio_bytes.rs      ← DirectIOBytesDataset
  prefetch.rs            ← Prefetch queue infrastructure (already bounded)
  sampler.rs             ← Sharding / worker splitting
```

The Python-facing API already works:
```python
loader = s3dlio.create_async_loader("s3://bucket/prefix/", {"prefetch": 16})
for item_bytes in loader:          # each item = one file's raw bytes today
    process(item_bytes)
```

**The extension**: add a new `ParquetRowGroupDataset` that implements `Dataset<Item=Bytes>` where **each item = one row group's selected column bytes** rather than one whole file. Once this exists, the current `create_async_loader` Python API works unchanged — the caller just adds `"format": "parquet"` to the options.

This approach:
- Re-uses all existing `DataLoader` prefetch, sharding, and concurrency infrastructure
- Eliminates Items #2 (bounded prefetch queue) and #3 (standalone Rust function) from the original plan — they're already handled
- Requires only two new files in s3dlio plus a small DLIO reader wrapper

---

## Revised Three-Item Plan

### Priority 1 (prerequisite): `ParquetRowGroupDataset` in s3dlio

**Where:** `s3dlio/src/data_loader/parquet_rg.rs` (new file)

**What it does:**
1. At construction: lists all `.parquet` files under the prefix
2. Fetches each file's Parquet footer once (range GET of last `footer_cap` bytes)  
3. Parses each footer using `ParquetMetaDataReader::decode_metadata(buf)` → `ParquetMetaData`  
4. Caches footer metadata in `Arc<Vec<ParquetMetaData>>` (never re-fetched during epoch)
5. Builds a flat index: `[(file_idx, rg_idx)]` of length = total RGs across all files

**`Dataset` implementation:**
```rust
impl Dataset for ParquetRowGroupDataset {
    type Item = Bytes;                    // raw merged column bytes for one row group

    fn len(&self) -> Option<usize> {
        Some(self.rg_index.len())         // 64 × 123 = 7,872 for DLRM
    }

    async fn get(&self, global_idx: usize) -> Result<Bytes, DatasetError> {
        let (file_idx, rg_idx) = self.rg_index[global_idx];
        let (start, length) = self.compute_extent(file_idx, rg_idx)?;  // pure CPU
        let uri = &self.file_uris[file_idx];
        self.store.get_range(uri, start, Some(length)).await
            .map_err(DatasetError::from)
    }
}
```

**`compute_extent` logic (pure CPU, using parquet crate):**
```rust
fn compute_extent(&self, file_idx: usize, rg_idx: usize) -> Result<(u64, u64)> {
    let meta = &self.file_metadata[file_idx];
    let rg = meta.row_group(rg_idx);
    let cols = self.col_indices_for_file(file_idx);     // defaults to all columns
    
    let mut start = u64::MAX;
    let mut end = 0u64;
    for ci in cols {
        let col = rg.column(ci);
        let col_start = col.dictionary_page_offset()
            .unwrap_or_else(|| col.data_page_offset()) as u64;
        let col_end = col_start + col.compressed_size() as u64;
        start = start.min(col_start);
        end = end.max(col_end);
    }
    Ok((start, end - start))
}
```

**Parquet crate API confirmed working (parquet 58.2.0):**
- `ParquetMetaDataReader::decode_metadata(buf: &[u8]) -> Result<ParquetMetaData>` — parse raw footer bytes  
- `meta.row_group(i)` → `RowGroupMetaData`  
- `rg.column(i)` → `ColumnChunkMetaData`  
- `.dictionary_page_offset()` → `Option<i64>`  
- `.data_page_offset()` → `i64`  
- `.compressed_size()` → `i64`  

**Footer byte range:** `size - footer_cap` to `size`. Use `footer_cap = 4 MiB` (covers all 64 DLRM files, avg footer = 2.66 MiB). Footer parsing via `ParquetMetaDataReader::decode_footer(slice: &[u8; 8])` gives the actual metadata length; full metadata is in `buf[..metadata_len]`.

**Cargo.toml addition (optional feature, no impact on existing builds):**
```toml
[features]
parquet = ["dep:parquet-crate"]

[dependencies]
parquet-crate = { package = "parquet", version = "58", default-features = false, optional = true }
```

**Minimal deps with `default-features = false`:** only thrift, bytes, chrono, half, hashbrown, num-bigint, num-traits. No arrow, no datafusion. Build time: ~14s.

---

### Priority 2: Python `create_async_loader` format routing

**Where:** `s3dlio/src/python_api/python_aiml_api.rs`

**Change:** In `create_async_loader`, detect `opts["format"] == "parquet"` and return a `PyBytesAsyncDataLoader` backed by `ParquetRowGroupDataset` instead of `S3BytesDataset`.

```rust
// In create_async_loader():
let format = opts.get("format").and_then(|v| v.extract::<String>().ok());
if format.as_deref() == Some("parquet") {
    let col_indices = parse_col_indices_from_opts(&opts)?;
    let footer_cap = opts.get("footer_cap").and_then(|v| v.extract::<usize>().ok())
        .unwrap_or(4 * 1024 * 1024);
    let dataset = ParquetRowGroupDataset::new(uri, col_indices, footer_cap, &loader_opts)?;
    return Ok(PyBytesAsyncDataLoader { dataset: PyDataset::ParquetRG(dataset), opts: loader_opts });
}
```

**Python API (unchanged from user perspective):**
```python
import s3dlio

loader = s3dlio.create_async_loader(
    "s3://mlp-flux/data/dlrm/train/",
    {
        "format": "parquet",
        "columns": list(range(40)),      # column indices to fetch
        "prefetch": 16,                  # RGs to keep in flight (uses existing DataLoader infra)
        "footer_cap": 4194304,           # 4 MiB footer window
    }
)

for rg_bytes in loader:
    # rg_bytes = raw merged column-chunk bytes for one row group
    # Parse with pyarrow once per RG (1,968 calls vs 16M)
    pass
```

**Note on `PyDataset` enum:** `python_aiml_api.rs` will need a new `ParquetRG` variant, or `ParquetRowGroupDataset` can be wrapped behind the existing `PyDataset`/`Arc<dyn …>` pattern — whichever is cleaner in context.

---

### Priority 3: DLIO reader wrapper

**Where:** `dlio_benchmark/dlio_benchmark/reader/parquet_reader_s3_rg.py` (new file)

**What it does:**
1. At `open(filename)`: calls `s3dlio.create_async_loader(prefix, parquet_opts)` — returns loader for ALL files
2. Yields one whole row group per Python call to the loader
3. Inner sample loop (Python `yield` for each row) uses `pyarrow` to parse the RG bytes once
4. Registered via `reader_factory.py` as format `parquet_rg`

**Call count comparison:**

| Operation | Current reader | New reader |
|-----------|---------------|-----------|
| `read_index()` Python calls | 16,000,000 | — (eliminated) |
| `next()` on RG loader | — | 7,872 |
| pyarrow parse per RG | 64 (open time) | 7,872 |
| Python `yield` per sample | 16,000,000 | 16,000,000 |
| `s3dlio.get_range()` calls | 1,968 | 0 (Rust internal) |

The Python `yield` per sample is unavoidable in a DLIO `Dataset.__getitem__`-style interface. However, with Priorities 1+2 implemented:

- **RG fetch overhead moves to Rust** — no Python involvement in extent calculation or scheduling
- **Prefetch depth controlled by `LoaderOptions.prefetch`** — no custom bounded queue needed
- **Sharding across workers handled by `LoaderOptions.worker_id / num_workers`** — no custom logic needed

**Throughput estimate after Priority 1+2+3:**

| Component | Time (per worker) |
|-----------|------------------|
| I/O: 1,968 RG fetches × 1.74 MiB @ 10 GiB/s | 0.33s |
| Rust overhead: extent compute + dispatch | <0.01s |
| Python: 16M yield calls × 0.5 µs | 8.0s |
| **Total** | **~8.3s → ~403 MiB/s per worker** |

Wait — the Python `yield` loop is still the wall. With 4 workers: **~1,612 MiB/s aggregate.**

**To break the 16M yield wall requires Priority 4 (see below).**

---

### Priority 4 (optional, maximum throughput): Arrow batch return from DataLoader

To escape the 16M Python-yield wall, the `ParquetRowGroupDataset` must return **structured batches** (Arrow RecordBatch) that PyTorch DataLoader can consume without per-row Python iteration.

**Architecture:**
```
ParquetRowGroupDataset::get(global_rg_idx) → RecordBatch (8,192 rows, 40 cols)
  ↓
Python sees: loader[rg_idx] = numpy array (8192, 40) via __array__
  ↓
DLIO reader: yields batch (8192 samples) per Python call → 1,968 calls total per worker
  ↓
PyTorch: sees 1,968 batches of 8192 rows = 16M samples total ✓
```

**Python call count with Priority 4:**

| Component | Time |
|-----------|------|
| I/O at 10 GiB/s | 0.33s |
| Python: 1,968 `next()` calls × 1 µs | 0.002s |
| NumPy/Arrow overhead per batch | ~0.01s |
| **Total** | **~0.34s → ~9,850 MiB/s per worker** |

**Tradeoff:** Requires adding `arrow` feature to s3dlio (`apache-arrow` crate, ~120 deps). Currently s3dlio has no arrow dependency — this was a deliberate architectural decision. Priority 4 should be a gated, separate `arrow-parquet` feature that does not affect existing builds.

---

## Implementation Sequence

```
Week 1:  Priorities 1+2 — ParquetRowGroupDataset in s3dlio
           → New src/data_loader/parquet_rg.rs
           → Cargo.toml: parquet feature flag
           → python_aiml_api.rs: format routing
           → build_pyo3.sh: add parquet to features
           → cargo test + cargo clippy (zero warnings)

Week 2:  Priority 3 — DLIO reader wrapper
           → dlio_benchmark/reader/parquet_reader_s3_rg.py
           → reader_factory.py registration
           → dlio config: format: parquet_rg
           → Benchmark: measure 1,612 MiB/s aggregate (4 workers)

Week 3+: Priority 4 (if needed) — Arrow batch return
           → Gated behind separate arrow-parquet feature
           → Benchmark: measure ~9,850 MiB/s per worker
```

---

## What the DataLoader Extension Replaces

Comparing original plan vs. revised plan:

| Original plan item | Revised plan | Reason |
|-------------------|-------------|--------|
| Item #2: Bounded prefetch queue (custom) | **Eliminated** | `LoaderOptions.prefetch` + `prefetch.rs` already handles this |
| Item #3: standalone `read_parquet_rg_extent()` fn | **Replaced by `ParquetRowGroupDataset`** | Dataset trait plugs into existing DataLoader, sampler, sharding infra |
| Item #1: IterableDataset (Python) | **Becomes Priority 3 wrapper** | Thinner — just wraps `create_async_loader`, no custom threading |

Net change: Less code, better architecture. All prefetch scheduling, concurrency control, and worker sharding re-use battle-tested s3dlio infrastructure instead of being re-implemented in Python.

---

## Files to Create/Modify

**s3dlio (Priority 1+2):**
- **NEW** `src/data_loader/parquet_rg.rs` — `ParquetRowGroupDataset`
- **MODIFY** `src/data_loader/mod.rs` — add `pub mod parquet_rg`; re-export
- **MODIFY** `Cargo.toml` — add `parquet` optional dep + feature flag
- **MODIFY** `src/python_api/python_aiml_api.rs` — format routing in `create_async_loader`
- **MODIFY** `build_pyo3.sh` — add `--features parquet` when building with parquet support

**dlio_benchmark (Priority 3):**
- **NEW** `dlio_benchmark/reader/parquet_reader_s3_rg.py` — DLIO reader wrapper
- **MODIFY** `dlio_benchmark/reader/reader_factory.py` — register `parquet_rg` format

**Configs:**
- New DLIO config `mlp-storage/configs/dlrm_parquet_rg.yaml` with `format: parquet_rg`

---

## Constraints

- `parquet` crate: always `default-features = false` (avoids arrow/datafusion pull-in)
- s3dlio: new code behind `parquet` feature flag — existing `cargo build` unaffected
- Zero warnings policy: `cargo clippy` must pass before any benchmark run
- Do NOT git commit/push without explicit user approval
- Do NOT modify other projects (dl-driver, sai3-bench, warpio) during this work
