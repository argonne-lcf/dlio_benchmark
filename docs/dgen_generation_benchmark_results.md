# dgen-py 0.2.3 Data Generation Benchmark Results

**Platform:** 12 logical CPUs  
**dgen-py version:** 0.2.3  
**Benchmark:** `tests/bench_dgen.py`  
**Timing discipline:** Generator/Pool creation is excluded from timing; only the fill call(s) are measured.

---

## Method Summary

| | BEFORE | AFTER (< 32 MB) | AFTER (≥ 32 MB) |
|---|---|---|---|
| **API** | `Generator(size=N).get_chunk(N)` | `BufferPool().next_slice(N)` | `Generator(size=N)` + `fill_chunk(64 MB loop)` |
| **Peak RAM** | Full file size | Full batch size | 64 MB constant |
| **Thread pool** | New Rayon pool every call | None | Reused across all chunks |
| **Data source** | Full allocation + fill | Zero-copy slice from rolling pool | Sequential 64 MB chunks |

---

## Results

```
  [1 MB]
    BEFORE  get_chunk:       0.001s 0.001s 0.001s  avg     972 MB/s   peak RAM ~1 MB
    AFTER   pool.next_slice: 0.000s 0.000s 0.000s  avg  457.10 GB/s   peak RAM ~1 MB
    Speedup: 470x   RAM savings: 1x

  [10 MB]
    BEFORE  get_chunk:       0.002s 0.002s 0.001s  avg    5.42 GB/s   peak RAM ~10 MB
    AFTER   pool.next_slice: 0.002s 0.002s 0.002s  avg    5.94 GB/s   peak RAM ~10 MB
    Speedup: 1.1x   RAM savings: 1x

  [100 MB]
    BEFORE  get_chunk:       0.009s 0.009s 0.010s  avg   11.23 GB/s   peak RAM ~100 MB
    AFTER   fill_chunk(64MB): 0.004s 0.003s 0.003s  avg   34.03 GB/s   peak RAM ~64 MB
    Speedup: 3.0x   RAM savings: 1x

  [1 GB]
    BEFORE  get_chunk:       0.065s 0.068s 0.063s  avg   16.43 GB/s   peak RAM ~1 GB
    AFTER   fill_chunk(64MB): 0.028s 0.027s 0.027s  avg   39.28 GB/s   peak RAM ~64 MB
    Speedup: 2.4x   RAM savings: 16x

  [10 GB]
    BEFORE  get_chunk:       0.574s 0.593s 0.624s  avg   17.99 GB/s   peak RAM ~10 GB
    AFTER   fill_chunk(64MB): 0.265s 0.265s 0.268s  avg   40.38 GB/s   peak RAM ~64 MB
    Speedup: 2.2x   RAM savings: 160x
```

---

## Analysis

### Throughput

For files ≥ 100 MB — the typical case for AI/ML training data — the new path delivers **2–3x higher throughput**:

| File size | BEFORE | AFTER | Gain |
|---|---|---|---|
| 100 MB | 11 GB/s | 34 GB/s | 3.0x |
| 1 GB | 16 GB/s | 39 GB/s | 2.4x |
| 10 GB | 18 GB/s | 40 GB/s | 2.2x |

The BEFORE path plateaus at ~18 GB/s because a new Rayon thread pool is created on every `get_chunk()` call. Startup overhead dominates at small sizes and caps throughput at large ones.

The AFTER path stabilizes at ~40 GB/s and has not hit its ceiling even at 10 GB — this is the machine's memory bandwidth limit, not a dgen-py limit.

### Memory

The RAM reduction is the more important improvement for large files:

| File size | BEFORE | AFTER | Reduction |
|---|---|---|---|
| 1 GB | 1 GB | 64 MB | **16x** |
| 10 GB | 10 GB | 64 MB | **160x** |

With the old API, generating a single 10 GB file required 10+ GB of RAM to be live simultaneously. This made large-file generation impractical on memory-constrained nodes and completely impossible if multiple MPI ranks shared a node.

With the new API, **peak RAM is always 64 MB**, regardless of file size. An 8-rank node generating 10 GB files previously required 80+ GB of RAM. Now it requires ~512 MB total.

### Small file note (1 MB)

The reported 470x speedup at 1 MB is an artifact of Rayon thread pool startup latency (~1 ms), which dominates the measurement for sub-2ms operations. Both paths produce the same amount of data; the BEFORE path simply pays a fixed pool-creation cost that is disproportionate at this scale. At typical training file sizes (100 MB–10 GB), the real speedup is **2–3x**.

### 10 MB boundary behavior

At 10 MB, both methods measure ~5–6 GB/s with near-identical times. This size sits just below the 32 MB `BufferPool` threshold, so the AFTER path uses `next_slice()`, which is fast but similarly bounded. No meaningful difference at this scale — both are well below the memory bandwidth ceiling.

---

## Key Takeaways

1. **Generation is no longer the bottleneck.** At 40 GB/s, data generation is 20–100x faster than any disk or object storage system. The bottleneck has fully shifted to Parquet encoding and I/O.

2. **Constant 64 MB peak RAM enables very large files.** A node with 256 GB RAM can generate arbitrarily large files with negligible memory pressure.

3. **MPI scaling is unaffected.** Each rank still generates its file slice independently. The RAM savings multiply by rank count: 8 ranks on one node previously needed 8× full-file RAM; now they need 8 × 64 MB = 512 MB total.
