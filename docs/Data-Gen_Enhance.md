# dgen-py Data Generation: Performance Analysis & Recommendations

> Analysis of `dgen-py` calling patterns in the DLIO benchmark's JPEG/PNG data generators,
> with benchmark results and a recommended change to reach ≥145K objects/second.

---

## Background: How `dgen-py` Generates Data Internally

`DataGenerator` (the Rust core behind `dgen_py.Generator`) produces data in **1 MB blocks**
(the minimum block size, clamped from any user-specified value).  Every `fill_chunk(n)` call
maps the requested byte range onto one or more 1 MB blocks and runs Xoshiro256++ independently
on each block.

This means object size relative to 1 MB **matters**:

- If the object fits entirely within one block → 1 MB of generation work, `n` bytes copied out.
- If the object straddles a block boundary → 2 × 1 MB of generation work, `n` bytes copied out.

---

## "Parallel Reused" vs "Parallel Fresh" — Root Cause

Both patterns use **sequential Xoshiro256++** (`max_threads=1`).  The performance difference
is purely due to block boundary alignment at 315 KB objects on a 1 MB block floor.

### Fresh — `DataGenerator(size=315KB)` created per call

Each call constructs a new `DataGenerator` with `size=315KB`.  `current_pos` always starts
at 0.  The 315 KB request fits entirely in block 0.

→ **1 block generated per call** (1 MB work for 315 KB output)

### Reused — `Generator(size=256GiB)` streaming continuously

`current_pos` accumulates across calls.  Because 315 KB does not divide 1 MB evenly
(GCD = 1 KB, LCM = 315 MB), roughly 31% of `fill_chunk` calls straddle a 1 MB boundary —
requiring **2 blocks** to serve 315 KB.  The remaining 69% require 1 block.

→ **1.31 blocks amortized per call** (31% overhead vs fresh)

### Results (28-core Xeon, 315 KB objects)

| Pattern | Blocks/call (amortized) | Objects/sec (per process) |
|:--------|:-----------------------:|:-------------------------:|
| Fresh — new `DataGenerator` per call | 1.00 | ~145K |
| Reused — `Generator` streaming | 1.31 | ~115K |
| **`BufferPool.next_slice()`** | **0.30** | **≥145K** |

---

## Current Code in `jpeg_generator.py` and `png_generator.py`

Both generators use the same streaming pattern:

```python
# Created once before the file loop (object-store non-DALI path)
_stream = _dgen_py.Generator(size=256 * 1024**3)   # 256 GiB, no block_size set
_buf    = bytearray(nbytes)                         # pre-allocated, reused

# Inside the per-file _write() closure:
mv = memoryview(_buf)[:nbytes]
_stream.fill_chunk(mv)   # fills from current_pos; ~31% of calls generate 2x 1 MB blocks
output.write(mv)
```

This is the **"reused" pattern** — ~115K objects/second at 315 KB.  The 31% overhead comes
from block boundary crossings that occur as `current_pos` advances through the 256 GiB stream.

---

## Recommended Fix: `BufferPool.next_slice()`

`BufferPool` generates one 1 MB backing block and serves **zero-copy Arc slices** from it.
At 315 KB, ~3.3 slices are served before the block is exhausted and a new one generated.
Amortized cost: **0.30 blocks/call** — the lowest of any pattern.

Replace in both `jpeg_generator.py` and `png_generator.py`, object-store non-DALI path:

```python
# Before the file loop — replace _stream + _buf with a single pool:
_pool = _dgen_py.BufferPool() if _HAS_DGEN else None

# Inside _write(), object-store non-DALI path:
if _pool is not None and not is_local:
    output.write(memoryview(_pool.next_slice(nbytes)))
```

Benefits over the current `Generator + bytearray + fill_chunk` pattern:

- **No pre-allocated buffer needed** — `next_slice()` returns a `BytesView` directly
- **Zero-copy** — `memoryview()` on the returned slice uses the buffer protocol with no copy
- **No block boundary overhead** — the 1 MB backing block is shared across ~3.3 objects
- **Simpler code** — two lines replace four

### Why not just use "fresh" `DataGenerator` per call?

Fresh avoids boundary crossings (1.00 block/call vs 1.31) and reaches ~145K obj/s, but it
**allocates and initialises a new `DataGenerator` struct on every call** — including PRNG
seeding from `urandom`.  `BufferPool` reaches the same throughput without per-call allocation
overhead and is the canonical pattern for sub-1 MB workloads.

> **Note**: `generate_buffer(size)` for `size < 1 MB` already uses a thread-local
> `BufferPool` automatically, so single-file scripts calling `generate_buffer()` in a loop
> already get this optimization without any changes.

---

## Multi-Process Scaling Reference (28-core Xeon)

For context on how dgen-py scales across independent Python processes (e.g. DLIO DataLoader
workers), each using `max_threads=1`:

| N processes (8 MB objects, 5 s run) | Aggregate GB/s | Per-process GB/s |
|:------------------------------------:|:--------------:|:----------------:|
| 1                                    | 5.2            | 5.2              |
| 4                                    | 17.8           | 4.5              |
| 8                                    | 27.9           | 3.5              |
| 16                                   | 52.7           | 3.3              |
| 28                                   | **58.6**       | 2.1              |

Aggregate throughput saturates DRAM bandwidth (~58 GB/s) at 28 processes.  The per-process
decline at high N is DRAM write saturation, not generator overhead.
