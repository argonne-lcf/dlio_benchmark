# Page Cache Flush at Epoch Boundary — ToDo

## Summary

Drop the Linux page cache at the start of each training epoch so that buffered file I/O reads
are always served from disk rather than from RAM on epoch 2+. This makes multi-epoch file
benchmarks accurate and comparable to object/O_DIRECT modes.

## Effect by I/O Mode

| Mode | Effect of `drop_caches` | Notes |
|------|------------------------|-------|
| `--file` (buffered) | **Useful** — forces real disk reads | Without it, epoch 2+ hit RAM, not disk |
| `direct://` (O_DIRECT) | Harmless, zero effect | O_DIRECT already bypasses page cache |
| Object / S3 | Harmless, zero effect | No local page cache involved |

Dropping unconditionally on every epoch is safe and correct for all modes.

## Constraint: Containers

`echo 3 > /proc/sys/vm/drop_caches` requires `CAP_SYS_ADMIN`.

- **Bare metal / root** (`--allow-run-as-root`): works fine.
- **Privileged container** (`--privileged` or `--cap-add SYS_ADMIN`): works fine.
- **Unprivileged container**: `open()` raises `PermissionError` (errno 13) or `OSError` — must be caught and silently ignored so the benchmark continues.

## Implementation Plan

**File**: `dlio_benchmark/dlio_benchmark/main.py`

**Location**: inside the epoch loop, rank 0 only, followed by a barrier.

```python
# Current code (~line 446):
for epoch in dft_ai.pipeline.epoch.iter(range(1, self.epochs + 1), include_iter=False):
    self.stats.start_epoch(epoch)
    ...
```

**Proposed change**:

```python
for epoch in dft_ai.pipeline.epoch.iter(range(1, self.epochs + 1), include_iter=False):
    if self.my_rank == 0:
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except OSError:
            pass  # silently skip in unprivileged containers or non-Linux
    self.comm.barrier()  # all ranks wait for rank 0 to finish flush before reading
    self.stats.start_epoch(epoch)
    ...
```

### Notes
- Only rank 0 writes to `/proc` — one writer is sufficient, avoids races.
- `self.comm.barrier()` ensures no rank starts reading until the flush is complete.
- `OSError` covers `PermissionError` (subclass), `FileNotFoundError` (non-Linux), and any other OS-level failure.
- `drop_caches = 3` drops pagecache + dentries + inodes (most thorough).
- No config flag needed: it is always safe and correct to flush before an epoch.
