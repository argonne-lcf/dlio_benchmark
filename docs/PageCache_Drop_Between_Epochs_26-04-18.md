# Page Cache Dropping Between Epochs — Why and How

**Date:** April 18, 2026  
**Status:** Proposal — documentation only, no code changes made

---

## Why This Matters

dlio_benchmark is a **storage** benchmark. Its measured throughput should reflect the performance of the storage system (NVMe, NFS, S3, parallel filesystem), not the client machine's DRAM.

On a local POSIX filesystem, the Linux kernel page cache automatically retains file data in DRAM after the first read. Subsequent reads of the same files are served from memory at 40–80 GB/s rather than from storage hardware. In a multi-epoch benchmark this means:

- **Epoch 1**: reads come from storage — accurate
- **Epoch 2+**: reads come from the page cache — measures DRAM, not storage

This is true for every format (NPZ, HDF5, Parquet, Arrow IPC), but the severity depends on the format. Parquet decode is CPU-intensive, which limits the effective read rate and naturally causes page eviction before the next epoch starts. Arrow IPC reads are trivial (memcpy only), so the OS can fully populate the cache and serve all subsequent epochs from DRAM. **Arrow IPC is more accurate for storage benchmarking in principle, but more vulnerable to page cache pollution in practice if the dataset fits in RAM.**

dlio_benchmark already detects this risk and logs a warning:
```
WARNING: The amount of dataset is smaller than the host memory; data might be
cached after the first epoch. Increase the size of dataset to eliminate the caching effect!
```

This warning is correct but passive. Dropping the page cache between epochs is a stronger, active solution that works regardless of dataset size.

---

## What Dropping the Page Cache Does

On Linux, writing `3` to `/proc/sys/vm/drop_caches` instructs the kernel to evict all clean pages (page cache, dentries, and inodes) from memory. This is a supported, non-destructive kernel interface — it only releases clean pages that can be re-read from storage; dirty pages (unsaved writes) are never evicted. The operation is instantaneous for moderate dataset sizes.

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
```

The `sync` flushes any pending writes before the drop, ensuring no dirty data is discarded. `sudo` is required because the operation is privileged.

After the drop, every read in the next epoch must come from storage hardware, making all epochs equally accurate.

---

## Why dlio_benchmark Should Do This Automatically

Currently there is no mechanism in dlio_benchmark to drop the page cache between epochs. The benchmark operator must do it manually before each run, which:

1. Is error-prone (easy to forget between runs)
2. Does not help for multi-epoch runs within a single invocation
3. Requires a separate out-of-band script or `--pre-run` hook

Building it into the benchmark ensures consistent, reproducible results across all epochs, not just epoch 1.

---

## Required Code Changes

Three files require modification. No new dependencies are introduced — only the Linux `/proc/sys/vm/drop_caches` interface is used, which is always available on Linux kernels 2.6.16+.

### 1. `dlio_benchmark/utils/config.py` — new config field

Add one boolean field to the `ConfigArguments` dataclass, near `odirect`:

```python
# In ConfigArguments dataclass, near odirect: bool = False
drop_page_cache: bool = False
```

And parse it from YAML in the `ConfigArguments.initialize_args()` / hydra config section (near where `odirect` is parsed):

```python
# In the reader config parsing block
if 'drop_page_cache' in reader:
    args.drop_page_cache = reader['drop_page_cache']
```

---

### 2. `dlio_benchmark/main.py` — cache drop helper function and epoch hook

Add a module-level helper function (alongside the existing `_apply_settle_guard`):

```python
def _drop_page_cache(args, comm) -> None:
    """Drop the Linux page cache between epochs for accurate storage benchmarking.

    Only activates when ALL of the following are true:
      - ``args.drop_page_cache`` is True
      - ``args.storage_type`` is LOCAL_FS or NFS (object storage is always cache-bypass)
      - The process is running on Linux
      - Rank 0 has write permission to /proc/sys/vm/drop_caches

    Rank 0 performs the drop; all ranks then barrier so they proceed together.
    If the write fails (not root, or not Linux), a warning is logged and
    the benchmark continues normally — the setting is advisory, not fatal.
    """
    if not args.drop_page_cache:
        return
    if args.storage_type not in (_StorageType.LOCAL_FS,):
        return   # object storage is inherently cache-bypass
    if args.my_rank == 0:
        try:
            # sync first to flush dirty pages (safety measure)
            import subprocess
            subprocess.run(['sync'], check=True)
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            args.logger.info(f"{utcnow()} Page cache dropped before epoch")
        except (OSError, PermissionError) as exc:
            args.logger.warning(
                f"{utcnow()} drop_page_cache=True but could not write to "
                f"/proc/sys/vm/drop_caches: {exc}. "
                f"Run as root or grant CAP_SYS_ADMIN, or use 'odirect: true' instead."
            )
    comm.barrier()
```

Call this function in `run()` inside the epoch loop, **after** `finalize()` (which closes open file handles) and **before** `reconfigure()` / the next epoch's reads:

```python
# In run(), inside the epoch loop — current code:
self.framework.get_loader(DatasetType.TRAIN).finalize()
if self.do_eval and epoch >= next_eval_epoch:
    ...
    self.framework.get_loader(DatasetType.VALID).finalize()
self.args.reconfigure(epoch + 1)
self.stats.end_epoch(epoch)

# Insert the cache drop here, after finalize() and before reconfigure():
self.framework.get_loader(DatasetType.TRAIN).finalize()
if self.do_eval and epoch >= next_eval_epoch:
    ...
    self.framework.get_loader(DatasetType.VALID).finalize()
_drop_page_cache(self.args, self.comm)   # ← new line
self.args.reconfigure(epoch + 1)
self.stats.end_epoch(epoch)
```

The placement is important:
- **After** `finalize()` — ensures all file handles are closed so no pages are pinned
- **Before** `reconfigure()` — ensures the cache is clean before the framework reinitialises its file list for the next epoch
- **Outside** the timing window — `stats.end_epoch()` / `stats.start_epoch()` bracket the actual I/O; the cache drop occurs between them and does not inflate measured throughput

---

### 3. YAML configuration

```yaml
reader:
  odirect: false         # O_DIRECT (bypasses cache entirely — preferred)
  drop_page_cache: true  # Drop page cache between epochs (requires root / CAP_SYS_ADMIN)
```

The two settings are complementary, not mutually exclusive:
- `odirect: true` — best accuracy, never populates cache, works without root if the device supports it
- `drop_page_cache: true` — clears cache accumulated during the epoch, requires root but works with any format and any reader
- Using both together is safe and provides belt-and-suspenders cache avoidance

---

## MPI / Distributed Considerations

In a multi-rank run, only **rank 0** should write to `/proc/sys/vm/drop_caches`. The drop is a local kernel operation — it only affects the machine it runs on. In a distributed job where multiple hosts each run ranks, only one rank per node should issue the drop. The implementation above uses `my_rank == 0` (global rank), which covers the single-host case. For multi-host MPI jobs, this should be extended to `local_rank == 0` (first rank on each node) using `DLIOMPI.get_instance().local_rank()` or equivalent.

---

## Relationship to the Existing `potential_caching` Warning

`StatsCounter` already computes `potential_caching` and logs a warning when `data_size_per_host_GB <= host_memory_GB`. The `drop_page_cache` feature complements this: rather than relying on the operator to manually increase dataset size, the benchmark can actively ensure cache-clean reads on every epoch. The warning should remain — it informs the operator about the risk — but `drop_page_cache: true` can be offered as the in-benchmark remedy.
