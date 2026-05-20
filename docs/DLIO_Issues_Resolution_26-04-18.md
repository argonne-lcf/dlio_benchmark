# DLIO Benchmark — Issue Resolution Summary

**Date:** April 18, 2026  
**Replaces:** `DLIO_IO_Issues-Proposal_2026-03-28.md`, `DLIO_IO_Issues-Executive_Summary_2026-03-28.md`, `DLIO_PR_Plan_26-04-13.md`, `DLIO_PR_Status-26-04-12.md`

---

## All Non-DALI Issues — Fully Resolved

| Issue | Description | Resolution |
|-------|-------------|------------|
| 1 / 6 | File vs object reader asymmetry; local readers incurring full CPU decode (PIL, NumPy, h5py) while S3 readers did not | `reader/_local_fs_iterable_mixin.py` added: parallel prefetch via `ThreadPoolExecutor`, byte count only, no decode. Affects `ImageReader`, `NPYReader`, `HDF5Reader`, `NPZReader`. |
| 2 | JPEG/PNG generator 300–1000× slower than necessary due to PIL encoding | `jpeg_generator.py` and `png_generator.py` now detect non-DALI loader and write raw bytes, skipping PIL encode. DALI path still produces valid encoded bitstreams. Confirmed: JPEG 3×, PNG 27× speedup. |
| 3 | TFRecord iterative sampler file-index bug: non-zero ranks read wrong files | `config.py` line 719: rank offset is now carried forward through all iterations via `my_rank * files_per_rank + sample_index // num_samples_per_file`. |
| 4 | `read_threads` hardcoded at 1, under-utilizing storage bandwidth | Auto-sized to `min(cpu_count // ranks_per_node, 8)` when user leaves default. Explicit YAML values respected. |
| 5 | Deduplication — files must be byte-unique | Already correct; `data_generator.py` uses `rng.integers(0, 2**63)` per file with `BASE_SEED + my_rank`. No code change needed. |
| 7 | 49 YAML config files with hardcoded lab IPs | External: mlp-storage now supplies endpoint/bucket/library via env vars and CLI overrides. Remaining S3 configs use `localhost` placeholder. |
| 8 | `multiprocessing_context` must be `spawn` for object-storage libraries | `config.py` auto-derives `spawn` when `storage_library` is `s3dlio` or `s3torchconnector`. Dataclass default changed from `fork` to `spawn`. |
| 9 | `storage_library` not wired to standard env vars; poor standalone usability | `config.py` `_apply_env_overrides()` now reads `DLIO_STORAGE_LIBRARY`, `DLIO_BUCKET`, `DLIO_STORAGE_TYPE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION`, and an optional `.env` file. |
| 10 / 11 | Data generation serial per rank; object-store uploads blocking | `data_generator.py`: seeds pre-derived in main thread (preserving determinism), then writes dispatched to `ThreadPoolExecutor`. New `write_threads` config field, auto-sized via `ranks_per_node`. |
| 12 | `comm_size` used as thread denominator — wrong on multi-node runs | `DLIOMPI.ranks_per_node()` added (`MPI_Allgather` of hostnames). `read_threads` and `write_threads` auto-sizing now uses `cpu_count // ranks_per_node`. |
| 13 | No settle-time guard after generation on eventual-consistency stores | `post_generation_settle_seconds: float = 0.0` field added. Non-zero + non-LOCAL_FS: rank 0 sleeps, then broadcasts barrier. Default 0.0 — no behavior change for existing configs. |
| 6b | `validate()` called before file lists available | Investigated: not a real bug. `derive_configurations()` does not call `validate()`; the only `validate()` call is in `main.py` after the file-list walk. Resolved with a clarifying comment in `data_generator.py`. |

---

## Outstanding Issues — DALI Only (Deferred: No GPU Available)

These were identified but not implemented because they require GPU hardware for validation:

| Issue | Description | State |
|-------|-------------|-------|
| DALI-1 | `shard_id` never passed in `dali_image_reader.py`, `dali_npy_reader.py`, `dali_tfrecord_reader.py` — all multi-rank DALI runs read shard 0 only | Branch `fix/dali-correctness` exists locally. Critical correctness bug — **do not use DALI with `comm_size > 1`** until fixed. |
| DALI-2 | `fn.python_function` callbacks re-introduce the GIL into DALI pipeline; full C++ JPEG decode is done and then discarded | Branch `feat/dali-modernization` exists locally. |
| DALI-3 | DALI 2.0 dynamic executor not adopted; `exec_dynamic=False` still in use | Deferred with DALI-2. |

---

## Minor Remaining Note

`storage_library` is still accessed via `(self.storage_options or {}).get("storage_library")` rather than as a first-class `ConfigArguments` dataclass field. Functionally correct — the env-var path from Issue 9 populates `storage_options['storage_library']` properly. The dataclass promotion (adding `storage_library: Optional[str] = None` directly) was not done; it is cosmetic and low risk.
