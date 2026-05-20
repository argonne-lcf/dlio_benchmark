# Batch API Design Analysis: GIL Overhead and Batch-Level Delivery

*Date: May 12, 2026*

---

## Context

`PyBytesAsyncDataLoader.items()` (s3dlio) returns a per-item synchronous iterator of
`PyObjectItem` values. Each `__next__()` call releases the GIL while waiting for the
next completed GET from Rust/Tokio's `buffer_unordered(N)` sliding window, then
reacquires it to build and return a `PyObjectItem { uri, data }`.

The question: **is GIL traversal overhead massive at ~1,640 completions/sec per worker,
and is a higher-level batch API worth building?**

---

## GIL Traversal Overhead — Is It Massive?

### Worker Architecture

DLIO's readers run inside **PyTorch `DataLoader` subprocess workers**
(`num_workers=read_threads`, with `multiprocessing_context`). Each worker is a
**separate OS process** with its **own independent GIL**. GIL *contention* across
workers is architecturally impossible — no two workers share the same interpreter.

### Cost Per GIL Crossing

A `__next__()` call from Python into a PyO3 `#[pyclass]` iterator involves:
- Releasing GIL: `py.allow_threads()` (releases the mutex)
- Blocking recv: `rx.blocking_recv()` (blocks OS thread, no CPU burn)
- Reacquiring GIL: mutex re-lock
- Building `PyObjectItem`: struct allocation + `uri: String` copy

Total cost: **~1–3 µs** per crossing on a modern CPU (cache-warm, no contention).

### Actual Crossing Rate

At s3-ultra loopback (8 µs latency, ~10 GB/s peak):
- 315 KB per object → ~39 µs per GET (transfer + latency)
- With 64 in-flight: **~1,640 completions/sec per worker**

GIL overhead per worker: `1,640 × 2 µs = 3.3 ms/sec = 0.33%`

### `get_sample()` Cost in DLIO

The image reader's `get_sample()` per item:
```python
def get_sample(self, filename, sample_index):
    byte_count = self._object_cache.get(filename, 0)   # dict lookup
    dlp.update(image_size=byte_count)                  # telemetry counter
    dft_ai.update(image_size=byte_count)               # telemetry counter
```
~1–5 µs total. At 1,640 items/sec: **5–13 ms/sec = 0.5–1.3%** overhead per worker.
All Python work per item (dict lookups, cache update, telemetry, batch counter) is
within this range.

### Bottom Line

**GIL overhead is real but not massive.** Total Python-side cost per worker: ~1–1.5%
of wall time. This will not show up as a hotspot in any profile. The concern is
understandable but the math does not support the "massive overhead" framing.

---

## Is a Batch-Level API Worth Building?

Yes — but for different reasons than GIL reduction.

### Reason 1: Reduced Python Object Churn

Per-item `items()` allocates per completion:
- One `PyObjectItem` Rust struct
- One Python `str` for `uri` (heap allocation, reference counted)
- One Python dict lookup round-trip

A `collect_batch(n)` returning `List[PyObjectItem]` amortizes allocator overhead:
Rust builds a `Vec<(String, Bytes)>`, converts it once to a Python list, and hands
it over in one GIL acquisition. Python iterates a plain Python list — no `__next__()`
dispatch per item.

### Reason 2: Path to True Zero-Copy Batch Delivery *(the genuinely interesting angle)*

For workloads with **fixed-size objects** (e.g., RetinaNet JPEGs at ~315 KB each),
Rust could:

1. Pre-allocate (or reuse) a contiguous `[u8; batch_size × max_file_size]` buffer
2. Fill it from `buffer_unordered(N)` as items complete — in-place, no copies
3. Wrap it as a numpy array via the buffer protocol
4. Return `(numpy_array, List[str] uris, List[int] actual_sizes)`

Python receives a `(batch_size, max_file_size)` numpy array with **zero Python-side
memory allocation and zero copying**. The batch is already in contiguous memory
ready for the decode pipeline. This is the path that no existing Python data loader
(PyTorch included) takes.

### Reason 3: Tail-Latency Concern (downside)

A `collect_batch(n)` approach requires Rust to wait for all `n` items to complete
before returning. With `buffer_unordered(64)` and `batch_size=16`, the batch
completion latency is driven by the **slowest of 16** concurrent GETs. At 8 µs
loopback variance is tiny, but at real S3 latencies (20–50 ms WAN), waiting for
the 16th item adds measurable P95 tail latency. The per-item `items()` approach
avoids this — Python sees items as they arrive.

### Reason 4: `num_samples_per_file` Alignment

DLIO's concept of a "batch" is `batch_size` **samples**, not files. For NPZ where
`num_samples_per_file=4`: 1 batch = 4 files. Rust only knows about files, not
sample structure within files. A file-level `collect_batch(n)` API requires the
caller to compute `n = batch_size // num_samples_per_file`. Manageable, but the
API must operate on files; Python converts files → samples.

---

## Design Options

### Option A — `collect_batch(n) -> List[PyObjectItem]` (simple, immediate)

Rust collects `n` items into a `Vec`, wraps as Python list. One GIL crossing.
Python iterates a native Python list — no iterator protocol overhead per item.
Easy to implement on top of the existing channel.

### Option B — `fill_buffer(numpy_array, n) -> List[str]` (zero-copy, advanced)

Python pre-allocates `numpy.empty((n, max_file_size), dtype=np.uint8)` once and
reuses it each batch. Rust writes fetched bytes directly into the numpy buffer
in-place via the buffer protocol. Returns the URI list. Zero Python-side memory
allocation per batch. Requires `max_file_size` known upfront.

### Option C — Keep `items()`, batch in Python (no Rust changes)

```python
import itertools
batch = list(itertools.islice(loader.items(), batch_size))
```

Still `n` GIL crossings — each `islice` element is a `__next__()` call. Does not
reduce overhead, just changes control flow structure.

---

## Recommendation

The GIL overhead at these rates with subprocess workers does **not** need fixing for
correctness or performance. If pursuing a batch API, the target should be
**Option B** — the zero-copy numpy path. That is the genuinely impactful optimization
for fixed-size-object workloads (RetinaNet, CosmoFlow), eliminating per-item Python
object creation and memory allocation entirely.

Option A is a good stepping stone toward Option B and easy to add without touching
the existing `items()` API.

Both live alongside `items()` — the per-item iterator remains the right primitive for
variable-size workloads (NPZ, HDF5, Parquet) where a buffer cannot be pre-sized.
