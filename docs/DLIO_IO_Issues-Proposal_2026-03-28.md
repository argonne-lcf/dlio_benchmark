# MPI Sharding & Parallelism Investigation: `dlio_benchmark`

**Date:** 2026-03-28

---

## 1. Data Generation

**File:** `dlio_benchmark/data_generator/data_generator.py`

**Sharding strategy — `_generate_files()`:**
```python
for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
    ...
```
Classic rank-stride sharding. Rank `r` owns files at global indices `r, r+comm_size, r+2*comm_size, …`. File paths are pre-computed in `self._file_list[i]`, which distributes them across `num_subfolders_train` round-robin. This is correct and reproducible.

**Seed handling:** `BASE_SEED + my_rank` for the per-rank RNG. File-level seeds are derived from a flowing `rng.integers(0, 2**63)` — no adjacent-seed correlation. Reproducible across runs.

**Directory creation bottleneck:** Only rank 0 creates directories (correct, but means all other ranks idle during the `create_node` loop for subfolders). On slow NFS with many subfolders, this is measurable latency.

**Intra-rank parallelism:** **None.** Each rank generates files in a serial loop. No threading. For large datasets on fast storage, each rank is I/O-bound writing one file at a time.

**Object store path:** After each file, `storage.put_data(path, bytes_value)` is called synchronously. No pipelining or async upload.

---

## 2. Data Loading (Training)

**Files:** `dlio_benchmark/utils/config.py` · `dlio_benchmark/data_loader/torch_data_loader.py` · `dlio_benchmark/reader/reader_handler.py`

### 2a. PyTorch path — `get_global_map_index()` (used when `data_loader_sampler == INDEX`)
```python
samples_per_proc = ceil(total_samples / comm_size)
start_sample = my_rank * samples_per_proc
end_sample   = (my_rank + 1) * samples_per_proc - 1
# ...
file_index = floor(global_sample_index / num_samples_per_file)
abs_path = file_list[file_index]
```
**Correct.** Each rank gets a contiguous slice of the global sample space. File-to-sample mapping is done via global index, so rank `r` naturally reads a contiguous block of files. The custom `dlio_sampler` pre-computes `[start_sample, end_sample]` and yields indices from that range.

Thread-level parallelism comes from `read_threads` (PyTorch `DataLoader` `num_workers`) with `multiprocessing_context` and `prefetch_factor`. Each worker independently reads samples.

### 2b. TF / iterative path — `build_sample_map_iter()` (used when `data_loader_sampler == ITERATIVE`)
```python
files_per_rank = (num_files // comm_size) % num_files
file_index = my_rank * files_per_rank          # ← initial offset
for sample in sample_list:                      # sample_list is global-indexed
    abs_path = file_list[file_index]
    sample_index += 1
    file_index = (sample_index // num_samples_per_file) % num_files  # ← LOCAL counter
```
**Bug:** The initial `file_index` (rank-aware offset) is applied only to the **first** sample. After that, `file_index` is driven by a LOCAL `sample_index` that starts at 0 regardless of rank. For rank 1 with `sppf=500`, rank 1 reads `file[2]` (correct), then immediately falls back to reading `file[0], file[0], …, file[1], …`.

This means the TF iterative path does **not** correctly shard files across ranks — it reads from mostly the wrong files for all non-rank-0 ranks. The PyTorch index path does not have this bug.

### 2c. Cross-rank file distribution pattern
All ranks share the **same flat global file list** built by rank 0 (via `storage.walk_node()` + sort). There is no per-rank subdirectory affinity. With `num_subfolders_train > 0` the files are distributed across subfolders, but each rank reads from any subfolder in the list — there is no "this rank owns this directory" concept.

---

## 3. Checkpointing

**Files:** `dlio_benchmark/checkpointing/base_checkpointing.py` · `dlio_benchmark/checkpointing/pytorch_checkpointing.py`

**Per-rank files:** Each checkpointing rank writes to `checkpoint_folder/global_epoch{E}_step{S}/model_states-{rank}.pt` independently — no rank serialization. Standard distributed checkpoint pattern.

**Who checkpoints:** Controlled by `zero_stage`, `tensor_parallelism`, `pipeline_parallelism`, and `data_parallelism`. With `zero_stage=0`, only ranks `< model_parallelism` actually write (data-parallel copies are deduplicated). This is correct.

**In-rank parallelism (checkpoint read):** `_get_streaming()` creates a `StreamingCheckpointing` instance with `num_parallel_readers=4`, `chunk_size=32MiB`. This parallelizes the read within a single rank's file. Writes happen via a single sequential stream.

**Memory model:** `_SizePlaceholder` (no actual tensor allocation) + `_compute_state_bytes()` → correct byte count passed to the streaming backend. No RAM proportional to model size is used during save/load.

**Barriers:** `comm.barrier()` after each checkpoint step in `_checkpoint_write()` / `_train()`. Optional `checkpoint_rank_sync` adds an extra barrier after every individual checkpoint. No barrier between individual layer writes within a rank.

**Layer writes are serial:** Within a rank, layers are saved in a `for layer_index in range(start_layer, end_layer+1)` loop — no threading across layers.

---

## 4. Summary Table

| Component | MPI Sharding | Intra-rank Threads | Key Issue |
|---|---|---|---|
| Data generator | ✅ stride `range(rank, N, size)` | ❌ None (serial) | No parallel file writes; slow for large datasets |
| Data loading (PyTorch) | ✅ contiguous sample slice, correct file mapping | ✅ `read_threads` workers | No per-rank directory affinity |
| Data loading (TF/iter) | ⚠️ Bug: only first file uses rank offset | ✅ `read_threads` | `build_sample_map_iter()` file_index resets to 0 after first sample |
| Checkpointing (write) | ✅ each rank writes its own file | ❌ layers written serially | No parallel layer writes per rank |
| Checkpointing (read) | ✅ each rank reads its own file | ✅ 4 parallel readers | Only parallelized on the read path |

---

## 5. Specific Improvement Opportunities

1. **Per-rank subdirectory ownership during generation and loading**: Set `num_subfolders_train = comm_size` and have rank `r` exclusively write to (and read from) `train/{r:04d}/`. This eliminates namespace contention on NFS/Lustre and makes the I/O pattern far more realistic for distributed storage. Today `num_subfolders_train` partitions files into folders but without rank affinity.

2. **Parallel intra-rank file generation**: Wrap the `_generate_files()` loop in a `ThreadPoolExecutor(max_workers=N)` — each thread writes an independent file (already uniquely seeded). This would N× generation throughput per rank on fast storage (NVMe, object store).

3. **Fix `build_sample_map_iter()` file index tracking**: The local `sample_index` counter should be replaced with the global sample index for the file lookup, matching the logic in `get_global_map_index()`. Currently rank 1+ in TF mode reads wrong files.

4. **Async object store upload**: In `_generate_files()`, the `storage.put_data(path, bytes)` call is synchronous. A bounded async queue (e.g., `asyncio` or `ThreadPoolExecutor`) would pipeline data generation and upload.

5. **Parallel checkpoint layer writes per rank**: The inner `for layer_index in range(start_layer, end_layer+1)` loop in `save_checkpoint()` is serial. Since each layer writes to an independent file, these could be parallelized with threads — especially relevant for large models with many layers.

6. **Read-ahead / file pinning**: The `read_threads` workers in PyTorch mode all operate on the global file list. Adding an optional `prefetch_list` derived from each rank's assigned file range (pinning files to DRAM via `mmap`) before training starts would eliminate open-file latency in tight training loops.

---

## 6. Additional Issues Identified on Second Review

### 6a. `storage_library` Promotion Inconsistency

**File:** `dlio_benchmark/utils/config.py` — `LoadConfig()` (line ~1075) and `validate()` (line ~368)

The YAML schema places `storage_library` as a top-level key under `storage:`:
```yaml
storage:
  storage_type: s3
  storage_library: s3dlio      # ← top-level in YAML
  storage_options:
    endpoint_url: https://...
```

But `validate()` reads it from inside `storage_options`:
```python
storage_library = (self.storage_options or {}).get("storage_library")
```

This only works because `LoadConfig()` performs an explicit "promotion" — it detects `config['storage']['storage_library']` and injects it into `args.storage_options['storage_library']`. So the YAML schema and the dataclass schema are inconsistent: `ConfigArguments` has no top-level `storage_library` field, and `validate()` assumes it has been promoted into `storage_options`.

**Risk:** Any code path that evaluates `storage_library` before or outside `LoadConfig()` (e.g., a custom runner that builds `ConfigArguments` by hand) will see `None`. Additionally, the Hydra CLI override path is ambiguous — both `++workload.storage.storage_library=s3dlio` (promoted by LoadConfig) and `++workload.storage.storage_options.storage_library=s3dlio` (direct) work, but neither is documented clearly, and users who pass the wrong one get an opaque `None` check failure.

**Recommendation:** Add `storage_library: str = ""` as a first-class field on `ConfigArguments`, read it directly in `validate()` from `self.storage_library`, and have `LoadConfig()` populate it without the promotion workaround.

### 6b. `validate()` Called Before File List Is Available

`derive_configurations()` (which calls `validate()`) is called twice:
1. During `DataGenerator.__init__()` with no file lists (the generator-only early path)
2. During `DLIOBenchmark.initialize()` after the storage walk

On the first call, credential checks and `storage_library` validation run even when the run is a pure file-system operation. More importantly, some validation branches (e.g., NPZ reader import checks) are exercised before it is clear whether object storage will actually be used. This is harmless when it works but adds unnecessary error surface for misconfigured environments.

**Recommendation:** Separate `validate_storage()` (called early, storage-type-aware) from `validate_workload()` (called after file lists are known). Only run credential checks when `storage_type == StorageType.S3`.

### 6c. `multiprocessing_context` Couples to `storage_library` But Lives in `reader:`

The `multiprocessing_context` key lives under `reader:` but its correct value depends entirely on the storage backend:

| Storage library | Required `multiprocessing_context` | Reason |
|---|---|---|
| `local_fs` / `minio` | `fork` (default) | No async runtime in worker |
| `s3dlio` | `spawn` | Tokio runtime destroyed by fork |
| `s3torchconnector` | `spawn` | Background S3 threads destroyed by fork |

This coupling is currently enforced only through comments in the YAML files. If a user copies a file-backend YAML and adds an s3dlio storage section without updating the reader section, all S3 reads will silently hang (the Tokio runtime is dead in the forked child). There is no runtime warning or error.

**Recommendation:** In `derive_configurations()`, after `storage_library` is known, automatically set `self.multiprocessing_context = "spawn"` if the library is `s3dlio` or `s3torchconnector`, with a warning if the YAML had explicitly set `fork`. This makes the constraint self-enforcing rather than documentation-dependent.

### 6d. Hardcoded Endpoint URIs in YAML Files

The lab IP `https://172.16.1.40:9000` appears hardcoded in every object-storage YAML:
```yaml
storage_options:
  endpoint_url: https://172.16.1.40:9000
```

This makes every object-storage YAML file **environment-specific** — they fail immediately in any other environment (CI, cloud, different lab). It also means the same model config cannot be shared across teams without edits.

**Recommendation:** Use environment variable resolution for all connection properties. Hydra supports `${oc.env:AWS_ENDPOINT_URL}` interpolation. Alternatively, treat `endpoint_url` as a required CLI override with no default, so the YAML template contains a clearly-marked placeholder:
```yaml
storage_options:
  endpoint_url: ???   # Required: set via ++workload.storage.storage_options.endpoint_url=...
```

### 6e. `build_sample_map_iter()` Bug — Concrete Description

For completeness, here is the exact effect of the file-index tracking bug (Section 2b) with a worked example:

Given 8 files, 2 ranks, 4 files per rank, `num_samples_per_file=1`:
- Rank 1: `files_per_rank = (8 // 2) % 8 = 4`. `file_index` starts at `1 * 4 = 4` (correct, pointing to file[4]).
- First iteration: reads `file[4]` ✅
- After first sample: `sample_index = 1`, `file_index = (1 // 1) % 8 = 1` — now pointing to `file[1]` ❌ (should be `file[5]`)
- All subsequent samples for rank 1 iterate through `file[1], file[2], file[3], …` — the same file range as rank 0.

Both ranks read overlapping files, meaning the benchmark double-counts throughput from the same data and misses the upper half of the dataset entirely. The PyTorch index path (`get_global_map_index()`) does not have this bug. TFRecord workloads using the `ITERATIVE` sampler are affected.

### 6f. No Barrier Before Directory Walk in `initialize()`

In `DLIOBenchmark.initialize()`, when `generate_data=True`, all ranks barrier after generation:
```python
self.data_generator.generate()
self.comm.barrier()   # ← correct
```

But then rank 0 immediately proceeds to `storage.walk_node()` inside the same `initialize()` call (after the barrier) to build `file_list_train`. On object stores with eventual-consistency semantics (or NFS with attribute caching), newly written objects may not yet be visible to a listing. There is no retry or wait logic before the walk. If the walk returns fewer files than expected, a hard exception is raised.

**Recommendation:** Add a configurable `post_generation_settle_time` (default 0) with a rank-0 sleep + broadcast before the walk when `storage_type != local_fs`.

---

## 7. YAML Config Proliferation Analysis

### 7a. Current State

The `configs/dlio/workload/` directory contains **49 YAML files** for what is effectively a small matrix of orthogonal dimensions:

| Dimension | Values |
|---|---|
| Model / workload | unet3d, resnet50, cosmoflow, llama3_8b, dlrm, flux, retinanet |
| Storage backend | local_fs, s3+minio, s3+s3dlio, s3+s3torchconnector |
| Phase | datagen only, train only, checkpoint only, train+checkpoint |
| Scale | a100, h100, b200, mi355, 1t, 405b, 70b, 8b |

The current approach creates one YAML per _combination_. For a single model (unet3d h100), this already produces 7 files:

```
unet3d_h100.yaml                 ← file, train
unet3d_h100_minio.yaml           ← minio, train
unet3d_h100_minio_datagen.yaml   ← minio, datagen
unet3d_h100_s3dlio.yaml          ← s3dlio, train
unet3d_h100_s3dlio_datagen.yaml  ← s3dlio, datagen
unet3d_h100_s3torch.yaml         ← s3torchconnector, train
unet3d_h100_s3torch_datagen.yaml ← s3torchconnector, datagen
```

Similarly, llama3_8b generates 4 files; cosmoflow, resnet50, flux, retinanet, dlrm create additional duplicates. This pattern scales as `O(models × libraries × phases)`.

### 7b. What Differs Between Files — and What Doesn't

Comparing the three unet3d-h100 training variants (minio / s3dlio / s3torch), **the only fields that differ** are:
```yaml
# Differs:
storage.storage_library:    minio | s3dlio | s3torchconnector
storage.storage_root:       mlp-minio | mlp-s3dlio | mlp-s3torch
storage_options.endpoint_url:  # same IP, but separate bucket implies separate data staging
reader.multiprocessing_context:  spawn  # same for all three object store variants
# s3dlio only:
storage_options.s3_force_path_style: true
# minio only:
storage_options.secure: false
```

**Everything else is identical**: model definition, framework, dataset sizes, record lengths, train epochs, computation time, batch size, read threads, shuffle settings, metric target AU.

The datagen variants differ from the train variants only in:
```yaml
workflow.generate_data: True   # vs False
workflow.train: False          # vs True
```

### 7c. Root Causes of the Explosion

1. **No config composition**: Hydra supports config groups (sub-directories with named YAML slices that can be composed), but the current setup uses a flat directory of monolithic files. There is no `defaults:` list or group structure.

2. **Storage connection params are baked in**: The endpoint URL and bucket name are specific to a single lab, making every file non-portable. Portable configs require parameterizing these, which currently gets done by forking.

3. **`workflow.generate_data` / `workflow.train` are toggled by file, not CLI**: Users fork the YAML to change phase rather than passing `++workload.workflow.generate_data=True` on the command line.

4. **`storage_library` is not a CLI-first parameter**: The library choice (minio vs s3dlio vs s3torchconnector) is buried inside the YAML, requiring a separate file per library instead of a single override on the command line.

---

## 8. Proposed YAML Config Architecture

### 8a. Principle: Separate What Changes from What Doesn't

The YAML files should capture stable model/workload facts (architecture, dataset sizes, target AU, epoch count, computation time). Storage backend and connection details should be supplied at runtime via CLI overrides or a small environment-local override file.

### 8b. Recommended Directory Structure (Hydra Config Groups)

```
configs/dlio/
  config.yaml          ← top-level Hydra config with defaults list
  workload/
    models/            ← config group: model + dataset + training params
      unet3d_h100.yaml
      resnet50_a100.yaml
      cosmoflow_a100.yaml
      llama3_8b.yaml
      dlrm_b200.yaml
      flux_b200.yaml
      retinanet_b200.yaml
    storage/           ← config group: storage backend templates
      file.yaml        ← local_fs, no credentials required
      s3_minio.yaml    ← s3 + minio SDK, endpoint_url = ???
      s3_s3dlio.yaml   ← s3 + s3dlio, endpoint_url = ???
      s3_s3torch.yaml  ← s3 + s3torchconnector, endpoint_url = ???
    workflow/          ← config group: what phases to run
      train.yaml       ← generate_data: False,  train: True
      datagen.yaml     ← generate_data: True,   train: False
      checkpoint.yaml  ← generate_data: False,  train: False, checkpoint: True
      full.yaml        ← generate_data: True,   train: True
```

A model file (`models/unet3d_h100.yaml`) would contain only stable facts:
```yaml
# configs/dlio/workload/models/unet3d_h100.yaml
model:
  name: unet3d
  type: cnn
  model_size: 499153191

framework: pytorch

dataset:
  data_folder: test-run/unet3d     # relative path within bucket or filesystem root
  format: npz
  num_files_train: 168
  num_samples_per_file: 1
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
  record_length_bytes_resize: 2097152

reader:
  data_loader: pytorch
  batch_size: 7
  read_threads: 4
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 5
  computation_time: 0.323

checkpoint:
  checkpoint_folder: checkpoints/unet3d
  checkpoint_after_epoch: 5
  epochs_between_checkpoints: 2

metric:
  au: 0.90
```

A storage template (`storage/s3_s3dlio.yaml`) would contain backend facts with required fields explicitly marked:
```yaml
# configs/dlio/workload/storage/s3_s3dlio.yaml
storage:
  storage_type: s3
  storage_library: s3dlio
  storage_root: ???                  # Required: bucket name, set via CLI
  storage_options:
    endpoint_url: ???                # Required: set via ++workload.storage.storage_options.endpoint_url=
    region: us-east-1
    s3_force_path_style: true

reader:
  multiprocessing_context: spawn     # Required for s3dlio — Tokio is fork-unsafe
```

### 8c. Command-Line Patterns for Runtime Switching

With this structure, switching backends requires only CLI overrides — no new YAML files:

**File-backend training:**
```bash
dlio_benchmark \
  workload=models/unet3d_h100 \
  ++workload.storage.storage_type=local_fs \
  ++workload.storage.storage_root=/mnt/scratch/dlio-data \
  ++workload.workflow.generate_data=False \
  ++workload.workflow.train=True
```

**Object storage with s3dlio:**
```bash
dlio_benchmark \
  workload=models/unet3d_h100 \
  ++workload.storage.storage_type=s3 \
  ++workload.storage.storage_library=s3dlio \
  ++workload.storage.storage_root=mlp-s3dlio \
  ++workload.storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} \
  ++workload.workflow.train=True
```

**Switch to minio on the same command, same model:**
```bash
# Change only storage_library and storage_root (bucket name)
... ++workload.storage.storage_library=minio \
    ++workload.storage.storage_root=mlp-minio \
    ++workload.reader.multiprocessing_context=fork
```

**Datagen-only, then train:**
```bash
# Step 1: generate
dlio_benchmark workload=models/unet3d_h100 \
  ++workload.storage.storage_type=s3 \
  ++workload.storage.storage_library=s3dlio \
  ++workload.storage.storage_root=mlp-s3dlio \
  ++workload.storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} \
  ++workload.workflow.generate_data=True \
  ++workload.workflow.train=False

# Step 2: train (identical flags, flip workflow)
dlio_benchmark workload=models/unet3d_h100 \
  ... \
  ++workload.workflow.generate_data=False \
  ++workload.workflow.train=True
```

### 8d. Environment-Local Override File (Alternative to Shell Functions)

For teams with a fixed endpoint, a local override file can be sourced by Hydra without committing credentials to the repo:

```yaml
# configs/dlio/local.yaml  (gitignored)
defaults:
  - override storage: s3_s3dlio

storage:
  storage_root: my-bucket
  storage_options:
    endpoint_url: https://my-minio.internal:9000
```

Then run:
```bash
dlio_benchmark +local=local workload=models/unet3d_h100 ++workload.workflow.train=True
```

### 8e. Impact on File Count

Under the proposed structure, the 7 unet3d-h100 files collapse to 1 model file plus 3 reusable storage templates (shared by all models). Across the full matrix of 7 models × 3 object libraries × 2 phases, the ~30 object-storage YAML files collapse to 7 model files + 3 storage templates + 3 workflow files = **13 files total** — a ~70% reduction, and all storage templates are shared across models.

### 8f. Short-Term Mitigation (No Refactor Required)

If the full Hydra config-group refactor is not immediately feasible, the proliferation can be stopped without changing existing files:

1. **Stop adding `_minio.yaml`, `_s3dlio.yaml`, `_s3torch.yaml` variants.** Document the override pattern in `README_S3DLIO_CONFIGS.md` instead.
2. **Remove hardcoded IPs** from existing YAML files. Replace with `???` (Hydra's "required, no default" sentinel) and add `endpoint_url` to the run instructions.
3. **Add a shared `storage/` config group** with the three library templates. New models only need a model YAML; storage is composed at runtime.
4. **Derive `multiprocessing_context`** automatically from `storage_library` in `derive_configurations()` to remove the hidden coupling.
5. **Phase switches via CLI**: Add one-line documentation showing `++workload.workflow.generate_data=True` so users stop forking YAML files to change only the phase.

---

## 9. The Core Principle: This Is a Storage Benchmark, Not a Data Processing Benchmark

### 9a. The Design Intent Is Correct — But the Implementation Is Incomplete

The codebase already recognises that decoded data has no value. In `reader_handler.py`, every read path ends with:

```python
# FormatReader.next()  and  FormatReader.read_index()
...
self.get_sample(filename, sample_index)   # reads + decodes file
self.preprocess()
return self._args.resized_image           # ← decoded data is THROWN AWAY here
```

`resized_image` is a **single random tensor**, allocated once at startup in `derive_configurations()`:

```python
self.resized_image = gen_random_tensor(
    shape=self.transformed_record_dims,
    dtype=self.transformed_record_element_dtype, rng=rng)
```

Every reader, every format, every sample in every epoch returns this same pre-allocated buffer. The content of what was read from storage is irrelevant by design. The benchmark measures how fast the storage can deliver bytes — not what those bytes mean.

The TFRecord reader already honours this principle fully: `_parse_image()` returns `self._resized_image` without touching the raw bytes at all. The S3 iterable readers (`image_reader_s3_iterable.py`, `hdf5_reader_s3_iterable.py`, `tfrecord_reader_s3_iterable.py`) store only byte counts for telemetry, never decoded arrays.

**The problem is that for local-filesystem readers and all generators, the code does substantial CPU-intensive data transformation work whose only output is a buffer that is immediately discarded.** Every CPU cycle spent on JPEG entropy coding, PIL decoding, protobuf serialization, or zlib compression is overhead injected into a storage benchmark that doesn't need it.

### 9b. Generator Overhead by Format

| Format | Generation work | Relevant to storage? | CPU cost |
|---|---|---|---|
| JPEG | `gen_random_tensor` → `PIL.fromarray` → `img.save(format='JPEG')` (DCT + quantize + Huffman) | ❌ | High: 10–60 ms/file |
| PNG | `gen_random_tensor` → `PIL.fromarray` → `img.save(format='PNG')` (Deflate lossless) | ❌ | Very high: 30–200 ms/file |
| NPY | `gen_random_tensor` (dgen-py) → `np.save()` (raw binary dump) | ✅ Near-minimal | Low: < 1 ms/file |
| NPZ (no compression) | `gen_random_tensor` → `np.savez()` (ZIP container, stored mode) | ✅ Near-minimal | Low |
| NPZ (zip compression) | `gen_random_tensor` → `np.savez_compressed()` (ZIP+Deflate) | ❌ | Medium–high: zlib per file |
| HDF5 (no compression) | `gen_random_tensor` → h5py metadata + raw dataset write | Mostly ✅ | Low–medium |
| HDF5 (gzip) | + GZIP compression per dataset | ❌ | Medium–high |
| TFRecord | `gen_random_tensor` → `tf.train.Example` → `SerializeToString()` per sample | ❌ partial | Medium: protobuf serialize |
| CSV | `gen_random_tensor` → `pd.DataFrame.to_csv()` (text encode + float formatting) | ❌ | Medium: text serialization |
| IndexedBinary | `gen_random_tensor` → MPI-IO raw byte write | ✅ Minimal | Low |
| Synthetic | single integer written as UTF-8 string | ✅ Minimal | Negligible |

**JPEG and PNG are the worst offenders** because the encoder is CPU-bound and irreversibly entangled in the format: there is no way to construct a valid JPEG or PNG without running the compression algorithm, because the file format *is* the compressed output.

### 9c. Reader Overhead by Format (Local Filesystem Path)

| Format | Reader `open()` / `get_sample()` work | Decoded data used? | CPU cost |
|---|---|---|---|
| JPEG/PNG (`ImageReader`) | `PIL.Image.open()` + `np.asarray()` — full entropy decode | ❌ Discarded | High: 5–20 ms/file |
| NPY (`NPYReader`) | `np.load()` — mmap or full array load | ❌ Discarded | Low–medium |
| NPZ (`NPZReader`) | `np.load()['x']` — ZIP inflate + array load | ❌ Discarded | Medium |
| HDF5 (`HDF5Reader`) | `h5py.File()` + `dataset[sample_index]` — HDF5 chunk read + numpy convert | ❌ Discarded | Low–medium |
| TFRecord (`TFReader`) | raw bytes streamed by tf.data, `_parse_image()` returns `resized_image` directly | ✅ Already bypassed | None |
| S3 iterable readers | raw bytes fetched, byte count stored for telemetry | ✅ Already bypassed | None |

The S3 iterable readers represent the correct pattern. They are documented explicitly:

> *"No PIL or numpy decode is performed. DLIO's FormatReader.next() yields a pre-allocated random tensor regardless of file contents; only the byte count is needed for the image_size telemetry metric."*
> — `image_reader_s3_iterable.py` docstring

The local-filesystem equivalents do not apply the same logic.

### 9d. Where Time Actually Goes in an End-to-End JPEG Benchmark Run

For a single 224×224 JPEG file on a local NFS filesystem:

**Generation (once):**
```
dgen_py random bytes:    ~0.01 ms   (fast Rust PRNG, zero-copy)
PIL.fromarray():         ~0.5 ms    (copies bytes into PIL Image object)
img.save(JPEG):         ~20–40 ms  (DCT + quantization + Huffman coding)
write() syscall:         ~0.1 ms    (kernel buffer, NFS async)
Total per file:         ~21–41 ms  — 98% is the JPEG encoder
```

**Reading (every training step, every epoch):**
```
open() syscall:          ~1 ms      (NFS RTT)
read() syscall:          ~0.01 ms   (115 KB at 10 GiB/s)
PIL.Image.open():        ~5–15 ms   (JPEG entropy decode + YCbCr→RGB)
np.asarray():            ~0.5 ms    (copy into numpy)
resized_image returned:  decoded array discarded
Total per file:         ~7–17 ms  — storage I/O is < 5% of total time
```

The storage benchmark is spending more time on JPEG decode during reading than on actual I/O. The encode during generation is 200–4000× the storage write time.

### 9e. The Non-Negotiable Constraint: Every File Must Contain Unique Bytes

Before discussing any optimisation, one constraint must be stated explicitly:

**Every generated file must contain content that is unique across the entire dataset. Reusing the same byte sequence across multiple files is a fundamental correctness error for a storage benchmark.**

Modern storage systems — enterprise NAS arrays (NetApp, Vast Data, Pure Storage), object stores, and distributed file systems — routinely apply inline deduplication and compression. If two files have identical byte content, a deduplicating storage system stores only one physical copy, regardless of how many logical files are created. A benchmark that writes N files containing identical bytes is not measuring how fast the storage can absorb N files of unique data — it is measuring how fast the dedup engine can detect and discard duplicates. The measured throughput may be orders of magnitude higher than true storage write throughput, producing completely meaningless results.

**The template-clone approach described in an earlier draft of this document was categorically wrong and has been withdrawn.** Writing the same pre-encoded JPEG bytes to every file would collapse 1.28 million "distinct" training images to a single unique 115 KB block in any deduplicating storage system. That is not a storage benchmark.

The same logic applies to any "pre-compute one serialized blob and copy it N times" shortcut for any format. The byte content of every file must be independently unique.

### 9f. dgen-py: The Correct Foundation for All Data Generation

The correct solution to the CPU overhead problem is already present in the codebase: `gen_random_tensor()` backed by **dgen-py**, a zero-copy Rust-backed PRNG library written specifically for this project.

Key properties that matter here:

- **Speed**: ~155× faster than NumPy random generation. For a 224×224×3 uint8 array (150,528 bytes), dgen-py generates the raw bytes in < 0.01 ms, versus ~1.5 ms for NumPy.
- **Uniqueness**: every call with a different seed produces a statistically independent, non-repeating byte stream. Since `_generate_files()` uses a flowing RNG that advances per file (`seed = int(rng.integers(0, 2**63))`), every file gets a unique seed → unique bytes.
- **Zero-copy**: dgen-py returns a `BytesView` implementing the buffer protocol. `np.frombuffer(bytesview, dtype=dtype)` consumes it without an intermediate allocation.
- **Scalability**: because the bytes are generated in Rust with SIMD, generation throughput exceeds 50 GiB/s on modern CPUs — faster than any storage device can accept data.

**dgen-py must be used for all new data generation, for all formats, without exception.** It is already wired into `gen_random_tensor()` and therefore already active for every format that calls it. The critical requirement is that no code path reuses byte content across file boundaries.

For the formats where generation work is proportional to storage size (NPY, IndexedBinary, HDF5 without compression), the pipeline is already correct:

```
dgen-py (unique bytes, < 0.01 ms per file) →  write() syscall to storage
```

dgen-py is the bottleneck only if the benchmark needs to generate faster than ~50 GiB/s per core, which exceeds every real storage system's ingestion bandwidth.

### 9g. JPEG/PNG: Do Files Need to Be ACTUALLY Valid Images?

The short answer: **it depends entirely on which data loader is configured.**

This is the key question for generation cost. If files do not need to be valid JPEG/PNG bitstreams, the generator can write raw dgen-py bytes directly — no PIL, no DCT, no Huffman coding — reducing generation from ~20–40 ms/file to < 0.01 ms/file. That is a 2000–4000× speedup.

#### When valid JPEG/PNG is required: DALI and NATIVE_DALI data loaders

`dali_image_reader.py` constructs a DALI pipeline that calls:

```python
images = fn.decoders.image(images, device='cpu')   # line 80
```

`fn.decoders.image()` is NVIDIA's GPU/CPU image decoder. It requires a syntactically valid JPEG or PNG bitstream. It will throw an error on random bytes, even if preceded by a correct-looking header. When `data_loader_type` is `dali` or `native_dali`, files MUST be valid images and PIL encoding is unavoidable.

#### When valid JPEG/PNG is NOT required: all other data loaders

The S3 iterable readers (`ImageReaderS3Iterable`) already prove this. They fetch raw bytes from object storage, record `len(raw_bytes)` for telemetry, and never call `PIL.Image.open()`. The benchmark runs correctly with files that contain arbitrary bytes — the format name attached to those bytes is irrelevant because the reader never decodes them.

After the Section 9h fix (replacing `PIL.Image.open()` with `open(filename, 'rb').read()` in the local-filesystem `ImageReader`), the same is true for all non-DALI paths:

| Data loader | Reader decodes image? | Files must be valid JPEG/PNG? |
|---|---|---|
| `pytorch` / `tensorflow` (local FS, current) | YES — `PIL.Image.open()` | YES (current) |
| `pytorch` / `tensorflow` (local FS, after 9h fix) | NO — raw byte read | **NO** |
| any (S3 iterable readers, already shipped) | NO — raw byte read | **NO** |
| `dali` / `native_dali` | YES — `fn.decoders.image()` | **YES, always** |

#### The consequence for generators: branch on `data_loader_type`

For non-DALI paths, `JPEGGenerator` and `PNGGenerator` can write raw dgen-py bytes directly, with no PIL pipeline at all:

```python
def _write(i, dim_, dim1, dim2, file_seed, rng, out_path_spec, is_local, output):
    if self._args.data_loader_type in (DataLoaderType.DALI, DataLoaderType.NATIVE_DALI):
        # DALI pipeline calls fn.decoders.image() — must produce valid JPEG
        records = gen_random_tensor(shape=(dim1, dim2), dtype=np.uint8, rng=rng)
        img = PIL.Image.fromarray(np.clip(records, 0, 255).astype(np.uint8))
        img.save(output, format='JPEG', quality=75)
    else:
        # Reader reads raw bytes and discards them — any bytes work
        raw = gen_random_tensor(shape=(dim1 * dim2 * 3,), dtype=np.uint8, rng=rng)
        output.write(raw.tobytes())
```

For the non-DALI branch the generation pipeline collapses to:

```
dgen-py (unique bytes, < 0.01 ms) → write() syscall to storage
```

This is identical to NPY generation. The "irreducible cost" of JPEG/PNG format disappears entirely for non-DALI configurations.

#### File size note

Raw dgen-py bytes for a 224×224×3 uint8 image = 150,528 bytes (~150 KB). A real JPEG of the same image is typically 50–115 KB (4:1–6:1 compression). The raw format produces slightly LARGER files than real JPEGs. For a storage benchmark, larger files per sample means more I/O per batch — a slightly more conservative (harder) test. This is acceptable. The `record_length` field in the benchmark config controls expected size; if exact size matching is needed, the raw write can be padded or truncated to `record_length` bytes.

#### Remaining mitigations for the DALI path

When `data_loader_type: dali` is configured, PIL encoding is unavoidable. The applicable mitigations are:

1. **Lower JPEG quality.** `quality=10` encodes at 3–5× speed compared to `quality=75`. Files are still valid, unique JPEG bitstreams.
2. **Parallel intra-rank encoding via `ThreadPoolExecutor`.** PIL's JPEG encoder releases the Python GIL; 4–8 threads per rank reduces wall-clock time proportionally.
3. **Use NPY or HDF5 for pure storage benchmarks.** DALI supports NPY input natively. If the goal is to measure storage bandwidth/IOPS rather than to simulate a specific vision training pipeline, switch formats. NPY generation is already fast and the benchmark result is equivalent.

**YAML warning recommendation for any JPEG/PNG config that uses `data_loader_type: dali`:**

```yaml
# WARNING: DALI data loader requires valid JPEG files (fn.decoders.image() is a real decoder).
# Generation cost: ~20-40ms/file (PIL JPEG encode). For faster generation with equivalent
# storage I/O measurement, use data_loader_type: pytorch with NPY format instead.
```

### 9h. Reader Overhead: The Fix That Is Already Half-Done

The S3 iterable readers already apply the correct pattern: fetch raw bytes, record the byte count for telemetry, discard the bytes, return `resized_image`. The local-filesystem `ImageReader` does not; it decodes the full JPEG via PIL.

The raw-byte-read fix for `ImageReader.open()` is valid and does not introduce any deduplication concern — the storage read is still a real read of the on-disk file (unique bytes are fetched); only the subsequent CPU decode is skipped:

```python
# Proposed replacement for ImageReader.open()
def open(self, filename):
    with open(filename, 'rb') as f:
        raw = f.read()
    return len(raw)   # byte count for telemetry, like ImageReaderS3Iterable

def get_sample(self, filename, sample_index):
    byte_count = self.open_file_map[filename]
    dlp.update(image_size=byte_count)
    dft_ai.update(image_size=byte_count)
```

This eliminates 5–20 ms of PIL decode overhead per sample from the training-step timing. The storage I/O — the thing being measured — is unchanged.

**NPZ/HDF5 with Compression:**
The same principle applies to read decompression. When `compression=gzip` or `compression=zip` is enabled, the reader spends significant CPU time inflating data that is then discarded. These settings should default to `none`:

```
WARNING at startup when compression != NONE:
"compression=<X> is enabled. Benchmark will include CPU decompression 
in timings, not pure storage bandwidth. Set compression=none for 
accurate storage performance measurement."
```

### 9i. Summary of Corrected Recommendations

| Issue | Correct Action | Incorrect Action (Do Not Do) |
|---|---|---|
| JPEG/PNG generation with non-DALI data loaders | Write raw dgen-py bytes directly — no PIL, no DCT, no Huffman; generation drops from ~30 ms/file to < 0.01 ms/file | Always run PIL encode regardless of whether the reader decodes the file |
| JPEG/PNG generation with DALI / NATIVE_DALI | PIL encode is unavoidable (`fn.decoders.image()` is a real GPU decoder); use `quality=10` + `ThreadPoolExecutor` | Treat DALI path the same as non-DALI and write raw bytes — DALI will throw an error on invalid bitstream |
| TFRecord per-sample protobuf serialization | Use dgen-py for each sample's raw bytes (already done); accept protobuf overhead as format cost | Pre-compute one `Example` blob and replicate it — produces N logically distinct but physically identical records |
| `ImageReader.open()` decodes JPEG to discard | Read raw bytes, store byte count (like `ImageReaderS3Iterable`) | Skip the storage read entirely — would produce an I/O-free benchmark |
| NPZ/HDF5 compression adds CPU overhead | Default `compression: none`; warn at startup when enabled | Add compression without warning — benchmark silently measures CPU, not storage |
| CSV format for storage benchmarking | Document as not recommended; prefer NPY/IndexedBinary | Add multi-format CSV confusion |
| JPEG/PNG for large-scale storage benchmarks with DALI | Document as "inherently generation-slow on DALI path"; recommend NPY/HDF5 for pure I/O testing | Use JPEG/PNG + DALI for billion-file benchmarks where generation time dominates |
| All data generation must use dgen-py | `gen_random_tensor()` via `_generate_files()` already does this — enforce as mandatory, no exceptions | Use `np.zeros`, `np.ones`, or any repeated constant — these produce identical content across files |

---

## 10. Small-File Workload Pathologies (JPEG / PNG)

### 10a. What "Small File" Means Here

JPEG and PNG formats always store exactly one sample per file (`num_samples_per_file = 1`). Typical sizes:

| Workload | Image size | File size |
|---|---|---|
| ImageNet-1K (resnet50) | 224 × 224 × 3 | ~50–150 KB |
| CIFAR-10 | 32 × 32 × 3 | ~2–5 KB |
| Custom satellite / medical | 512 × 512 × 1 | ~100–500 KB |

Unlike TFRecord, HDF5, or NPZ — which pack hundreds or thousands of samples into one file, amortising open/stat/read latency across many samples — every JPEG/PNG access is a full open → read → decode → close cycle for a single sample. This makes the number of IOPS required proportional to the sample count, not the batch count.

### 10b. Data Generation Bottleneck

`_generate_files()` in `data_generator.py` drives every format generator. Its core loop is:

```python
for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
    write_fn(i, dim_, dim1, dim2, ...)   # serial within rank
```

There is no thread pool, no `asyncio`, no `concurrent.futures`. Each call to `write_fn` must complete before the next begins.

For JPEG and PNG, `write_fn` is:

```python
# jpeg_generator.py
img = Image.fromarray(arr.astype('uint8'), mode='RGB')
img.save(output, format='JPEG')       # CPU-bound encode, ~10–60 ms

# png_generator.py
img = Image.fromarray(arr.astype('uint8'), mode='RGB')
img.save(output, format='PNG')        # CPU-bound lossless encode, ~30–200 ms
```

PIL's JPEG and PNG encoders are single-threaded inside each call. JPEG encode at quality 75 typically runs 15–40 ms for a 224×224×3 image on a modern core; PNG is 2–5× slower due to lossless compression.

**Concrete example — ImageNet-scale dataset (1.28 M files) with NP=8:**

| Metric | Value |
|---|---|
| Total files | 1,280,000 |
| Files per rank (`N / np`) | 160,000 |
| Encode time (JPEG, 30 ms/file) | 160,000 × 0.030 s ≈ **80 min per rank** |
| Encode time (PNG, 100 ms/file) | 160,000 × 0.100 s ≈ **4.4 hours per rank** |
| Storage write time (100 KB, 1 GiB/s NFS) | 160,000 × 0.0001 s ≈ **16 s** — negligible |

The bottleneck is not I/O bandwidth — it is pure CPU time for compression. Because each rank is serial, adding more MPI ranks scales generation linearly, but the per-rank CPU time remains unchanged. Doubling NP from 8 to 16 halves the wall-clock time, but only by adding 8 more processes. There is no intra-rank parallelism to exploit the spare CPU cores that sit idle while one thread encodes.

**Contrast with `hdf5_generator.py` and `npy_generator.py`:** NumPy native binary format saves raw memory-mapped arrays at speeds limited only by storage bandwidth (often 1–5 GiB/s per rank). JPEG/PNG generation is an order of magnitude slower for the same logical data volume.

### 10c. Data Reading Bottleneck

`image_reader.py` uses PIL to read files:

```python
def open(self, filename):
    # called once per sample, from a DataLoader worker process
    img = Image.open(filename)
    data = np.asarray(img)
    self.open_file_map[filename] = data
```

Each call is a separate system-level open → read → JPEG decode → numpy conversion. There is no read-ahead, no batch opening, and no memory pooling across calls.

**Throughput ceiling for `read_threads=1` (the default):**

On NFS (RTT ~1 ms, bandwidth ~10 GiB/s), each file fetch is dominated by per-request latency:

- Per-file time ≈ RTT + file_size/bandwidth = 1 ms + (115 KB / 10 GiB/s) ≈ 1.01 ms
- Maximum IOPS ≈ 990 files/sec
- Throughput ≈ 990 × 115 KB ≈ **114 MiB/s** — with 10 GiB/s of available bandwidth **98.9% idle**

With `read_threads=8`:

- 8 concurrent opens → 8 simultaneous RTTs → IOPS ≈ 7,920 → **912 MiB/s** — still only 9% of NFS bandwidth

With `read_threads=32`:

- 32 concurrent opens → IOPS ≈ 31,680 → **3.6 GiB/s** — 36% of NFS bandwidth

The practical takeaway: **IOPS, not bandwidth, is the binding constraint for small-file JPEG/PNG reading**. The optimal `read_threads` value is `ceil(target_throughput / (file_size / bandwidth) + RTT * target_IOPS)`, which for typical deployments means 16–64 threads per rank, not the default of 1.

### 10d. No Aggregated-Access Path

Frameworks such as WebDataset, FFCV, and TFRecord address this problem by grouping many samples into sequential tar or binary shards. A single large sequential read then yields many samples, converting the random-IOPS problem into a streaming-bandwidth problem. DLIO has no sharding path for JPEG or PNG: every benchmark run, at every scale, reads each sample as an individual file. This is by design for the benchmark (measuring actual per-file I/O cost), but it means:

1. Any benchmark result with JPEG/PNG and `read_threads` < 16 is almost certainly I/O starved, not representative of storage peak capability.
2. Results should always report `read_threads × comm_size` (total concurrent I/O streams) alongside throughput.

### 10e. Sub-folder Namespace

`num_subfolders_train` distributes files across sub-directories, reducing directory listing time on large NFS servers. It does not change the fundamental one-file-per-open access pattern. For datasets with > 100 K JPEG/PNG files, sub-folders are necessary to avoid NFS `readdir` stalls, but are not sufficient to close the throughput gap.

---

## 11. `read_threads` — Fixed YAML Value vs. Runtime-Adaptive Sizing

### 11a. Current Behaviour

`read_threads` is defined in `ConfigArguments`:

```python
read_threads: int = 1     # dlio_benchmark/utils/config.py
```

It is set at YAML-load time (before MPI is initialized) and passed verbatim to PyTorch `DataLoader(num_workers=read_threads)`. The only runtime check is in `validate()`:

```python
if self.read_threads > 1:
    cores_available = len(psutil.Process().cpu_affinity())
    if cores_available < self.read_threads:
        self.logger.warning(...)    # logs a warning, zero action taken
```

Validation checks the pinned CPU set of the current process, not the actual core count divided by ranks per node. It never modifies `read_threads`, caps it, or auto-computes a value. DLIOMPI's `npernode()` and `nnodes()` are never consulted from `config.py`.

The `prefetch_factor` fed to PyTorch DataLoader is:

```python
prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
```

This means that changing `read_threads` without correspondingly adjusting `prefetch_size` silently changes prefetch aggressiveness, which affects memory consumption and training-step hide latency in ways that are not visible in the YAML.

### 11b. Thread Budget Analysis Across Deployment Scales

When `read_threads = 8` is hardcoded (as in `resnet50_a100.yaml`), the total DataLoader worker processes per node is `read_threads × ranks_per_node`:

| Deployment | ranks/node | read_threads | DataLoader workers/node | Total processes/node | 128-core utilisation |
|---|---|---|---|---|---|
| NP=1, 1 node | 1 | 8 | 8 | 9 | 7% |
| NP=8, 1 node | 8 | 8 | 64 | 72 | 56% |
| NP=8, 8 nodes | 1 | 8 | 8 | 9 | 7% |
| NP=64, 8 nodes | 8 | 8 | 64 | 72 | 56% |

The same YAML sets the same thread count regardless of whether one or eight ranks share a node. On high-rank-density nodes (NP=8/node), `read_threads=8` allocates 64 reader processes per node and may saturate the NFS client connection pool or cause CPU thrashing. On single-rank nodes, `read_threads=8` leaves most cores idle while I/O is the bottleneck.

**The correct thread budget formula is:**

```
read_threads_per_rank = max(1, floor(available_cores / ranks_per_node / cpu_per_io_thread))
# For I/O-bound NFS: cpu_per_io_thread ≈ 0.5 (threads mostly sleep on syscalls)
# For CPU-bound JPEG decode: cpu_per_io_thread ≈ 1.0
# Practical range: [2, 64]
```

DLIOMPI can provide all the inputs (`npernode()`, via `MPI.COMM_TYPE_SHARED`), and `os.cpu_count()` or `psutil.cpu_count()` gives the core total. The computation is straightforward but requires MPI to be initialized before validation, which conflicts with the current order of operations (see Section 6b).

### 11c. The Fixed-vs-Auto Design Decision

**Arguments for keeping `read_threads` as a fixed YAML integer:**
- Reproducibility: same YAML, same thread count, same result regardless of hardware.
- Simplicity: no implicit logic; user controls the knob directly.
- Explicit: reported clearly in output logs.

**Arguments for auto-sizing:**
- The "correct" value differs by an order of magnitude between single-node and multi-node deployments of the same YAML.
- The default of 1 is severely under-threaded for any network storage workload.
- Users who do not know to raise `read_threads` will see misleadingly low throughput that is not representative of storage capability.

**Recommendation:** Support `read_threads: auto` as a special sentinel value. When set to `auto`, compute at runtime:

```python
import os
ppn = DLIOMPI.get_instance().npernode()
total_cores = os.cpu_count() or 8
# Reserve 1 core per MPI rank for compute; divide remainder among I/O threads
io_threads = max(1, min(64, (total_cores - ppn) // ppn))
self.read_threads = io_threads
```

Log the resolved value at the start of the run so it appears in benchmark results. Keep the integer form working unchanged for reproducible benchmark runs.

---

## 12. MPI Multi-Host Topology — Available Infrastructure, Missing Integration

### 12a. What DLIOMPI Already Tracks

`DLIOMPI.initialize()` uses `MPI.COMM_TYPE_SHARED` to discover per-node topology at startup:

```python
split_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
local_ppn = split_comm.size          # ranks sharing this node
self.mpi_local_rank = split_comm.rank
# Gather ppn across all nodes via leader communicator
self.mpi_ppn_list = COMM_WORLD.bcast(ppn_list, root=0)
self.mpi_nodes = len(self.mpi_ppn_list)
self.mpi_node  = <node index for this rank>
```

The public API is:

| Method | Returns |
|---|---|
| `rank()` | Global MPI rank (0…comm_size-1) |
| `size()` | Total MPI world size |
| `local_rank()` | Rank within this node (0…ppn-1) |
| `npernode()` | Ranks on this node (can vary per node) |
| `nnodes()` | Total node count |
| `node()` | Node index for this rank |

This is a complete node-topology picture. It is used in `statscounter.py` (for the benchmark summary) and in `base_checkpointing.py` (line 424: cross-node checkpoint read offset). It is **not used** in `data_generator.py` or `config.py`.

### 12b. Scaling Formulas as NP and HOST Vary

The training sample distribution is:

```
samples_per_proc = ceil(total_samples / comm_size)
training_steps   = ceil(total_samples / batch_size / comm_size)
```

where `comm_size = NP * HOST` (total ranks). These scale correctly with the product, but they contain no node-level term. The formulas do not distinguish between:

- 64 ranks on 1 node (NP=64, HOST=1): all ranks share the same NFS mount, causing ~64× connection multiplexing
- 64 ranks on 64 nodes (NP=1, HOST=64): each node has a dedicated NFS mount, maximally parallelising metadata operations

**For JPEG/PNG reading**, effective storage throughput scales as:

```
IOPS_total = ranks_total × read_threads × (1 / per_open_latency)
```

where `per_open_latency` includes NFS RTT, kernel VFS overhead, and JPEG decode time. This throughput grows with both axes (ranks and threads), but the per-node NFS mount bandwidth caps growth when all ranks share one mount. The benchmark currently cannot express or control which axis scales which way.

**Concrete scale-up table (JPEG, 115 KB/file, NFS RTT=1ms, BW=10 GiB/s/node):**

| NP | HOST | comm_size | read_threads | IOPS_total | Throughput |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 990 | 114 MiB/s |
| 4 | 1 | 4 | 8 | 15,840 | 1.8 GiB/s |
| 8 | 1 | 8 | 8 | 31,680 | 3.6 GiB/s → NFS BW cap (10 GiB/s single mount) |
| 4 | 8 | 32 | 8 | 126,720 | 14.6 GiB/s → 8 × NFS BW cap |
| 8 | 32 | 256 | 8 | 1,013,760 | 116 GiB/s |

The key insight: **scale-out across hosts is much more effective than adding ranks per node**, because each new host brings a fresh NFS connection budget and independent bandwidth. DLIO's fixed `read_threads` value in YAML does not guide the user toward this topology insight.

### 12c. File Distribution and Node Locality

Data generation currently assigns files via:

```python
for i in range(my_rank, total_files, comm_size):
    write_fn(file_list[i], ...)
```

This is a round-robin stride across the global rank space. With `num_subfolders_train > 1`, the file-to-subfolder assignment is:

```python
subfolder = i % num_subfolders_train
```

Both mappings are rank-indexed, not node-indexed. If `num_subfolders_train = num_nodes`, the intent might be to give each node its own subfolder for locality, but the actual assignment distributes files from all nodes into all subfolders (because `i % comm_size` spans all ranks, not just the ranks on one node). Ranks on node 0 produce files in all subfolders, as do ranks on node 1, etc.

For read locality on distributed file systems with per-directory locking (some NFS and Lustre configurations), concentrating each node's reads into its "own" subfolder can reduce contention. The current round-robin prevents this. A node-local assignment would be:

```python
node_idx = DLIOMPI.get_instance().node()
subfolder = node_idx % num_subfolders_train
```

This is not currently implemented.

### 12d. What Is Missing

| Gap | Current state | Impact |
|---|---|---|
| `read_threads` not scaled by `npernode()` | Hardcoded YAML integer | Over-commits per-node CPU when ranks/node is high; under-commits on single-rank nodes |
| No intra-rank generation parallelism | Serial `_generate_files()` loop | JPEG/PNG generation CPU-bottlenecked; idle cores cannot be exploited |
| Node-local file affinity not implemented | Round-robin across all ranks | No NFS namespace locality; all nodes contend on all subfolders |
| Benchmark output does not report `npernode()` | `num_hosts` reported, `ppn` not | Cannot reconstruct per-node concurrency from published benchmark results |
| `read_threads` is set before MPI init | Load-time YAML evaluation | Auto-sizing using `npernode()` requires a post-MPI-init resolve step |

### 12e. Recommendations

1. **Log MPI topology in benchmark header**: At rank 0, emit `nnodes()`, `npernode()`, and `read_threads` so that any published result has sufficient information to reproduce the I/O concurrency.

2. **Auto-size `read_threads` post-MPI-init**: If `read_threads: auto` (or `read_threads: 0` as a sentinel), resolve to `max(1, min(64, (os.cpu_count() - npernode()) // npernode()))` after `DLIOMPI.initialize()`. This requires moving the resolution step out of YAML parse and into `derive_configurations()`, which already runs inside the main process after MPI init.

3. **Add intra-rank concurrency for JPEG/PNG generation**: Wrap the `_generate_files()` loop in a `concurrent.futures.ThreadPoolExecutor`. PIL's JPEG encoder releases the GIL during its C extension work; threads genuinely parallelise the CPU encode. A pool of `min(read_threads, 8)` workers per rank would reduce ImageNet-scale generation from hours to minutes without requiring any MPI changes.

4. **Node-indexed subfolder assignment**: When `num_subfolders_train == nnodes()`, assign `subfolder = node()` per rank so that all reads for a given training step from one node hit one subfolder. This concentrates hot NFS metadata into per-node directories, reducing cross-node directory contention.

5. **Document the NP vs HOST scaling trade-off**: Add a section to the benchmark README explaining that for JPEG/PNG workloads, scaling HOST outperforms scaling NP for the same `comm_size`, because each new host brings independent NFS bandwidth. Provide a concrete example using the IOPS formula above.

---

## 13. File vs. Object Workload Asymmetry — Closing the Performance Gap

### 13a. The Problem: Two Classes of Benchmark with Different Overhead Profiles

The S3 iterable readers introduced for object storage were built with a correct understanding of DLIO's design principle: the benchmark measures storage throughput, not data transformation throughput. As a result, every S3 iterable reader — `ImageReaderS3Iterable`, `NPYReaderS3Iterable`, `HDF5ReaderS3Iterable`, `TFRecordReaderS3Iterable` — does the following:

1. Fetch raw bytes from the storage system (the I/O operation being measured).
2. Record the byte count for telemetry (`image_size` metric).
3. Return `self._args.resized_image` (the pre-allocated random tensor).
4. Never decode, decompress, or numpy-convert the fetched bytes.

The local-filesystem readers — `ImageReader`, `NPYReader`, `HDF5Reader` — do NOT apply this principle. `ImageReader` calls `PIL.Image.open()` and `np.asarray()` on every sample. `NPYReader` calls `np.load()`. `HDF5Reader` performs a full HDF5 chunk read and numpy conversion. All of this CPU work happens inside the training-step timing window, and all of it produces output that is immediately discarded.

**The result is that the same workload, with the same files, produces fundamentally different benchmark numbers depending solely on whether the storage backend is local FS or object storage.** An object-storage run with `ImageReaderS3Iterable` and a local-FS run with `ImageReader` are not measuring the same thing — even if the physical data is identical.

### 13b. Quantified Impact of the Asymmetry

For a JPEG workload at 224×224×3 image size, the per-sample overhead difference:

| Reader | Storage I/O time | CPU decode time | Total per sample | CPU fraction of total |
|---|---|---|---|---|
| `ImageReaderS3Iterable` (object) | ~1–5 ms net fetch | 0 ms | ~1–5 ms | 0% |
| `ImageReader` (local FS) | ~0.01 ms read | 5–20 ms PIL decode | ~5–21 ms | 71–99% |

A benchmark using `ImageReader` on a fast NVMe filesystem can show **5–20× lower per-sample throughput than a benchmark using `ImageReaderS3Iterable` on the same data served from an object store** — not because the object store is faster, but because the local-FS reader does far more CPU work. Published benchmark comparisons between the two backend types are therefore not valid without correcting for this asymmetry.

The same asymmetry exists at generation time: object store YAML configs typically target fewer total files or use NPY/HDF5 format (avoiding JPEG), while local FS YAML configs often use JPEG with no awareness of the PIL encode cost. This is an accident of how the configs evolved, not a deliberate design choice.

### 13c. Why the Asymmetry Exists

The object-store readers were written later, after the design principle (Section 9a) was understood. The local-filesystem readers predate that understanding and have not been updated. The S3 iterable reader docstrings explicitly document why decoding is wrong:

> *"Calling `PIL.Image.open(BytesIO(raw))` on JPEG/PNG data is pure CPU overhead. DLIO's `FormatReader.next()` yields a pre-allocated random tensor regardless of file contents; only the byte count is needed for the image_size telemetry metric."*

The same rationale applies to `ImageReader`, but that file contains no equivalent comment and no equivalent implementation. The optimization was applied to the new path and never back-ported to the original one.

For data generators, the object-store configs incidentally avoid the worst-case formats (JPEG/PNG with PIL encode) because they were configured for network-storage scale testing where generation cost is more visible. The local-FS configs retain JPEG/PNG as the default for historical reasons.

### 13d. The Rationalization Proposal

The fix is to bring local-filesystem readers up to the standard already established by the S3 iterable readers. This is a code change only — no format changes, no YAML changes, no protocol changes. The storage I/O (the measured operation) is unchanged in every case.

**Reader rationalization targets (by priority):**

| Reader | Current behaviour | Rationalized behaviour | Change required |
|---|---|---|---|
| `ImageReader` (local FS JPEG/PNG) | PIL decode + numpy convert | Raw byte read, byte count for telemetry | Replace `PIL.Image.open()` with `open(rb).read()` |
| `NPYReader` (local FS NPY/NPZ) | `np.load()` — allocates full array | Raw byte read, byte count for telemetry | Replace `np.load()` with `open(rb).read()` |
| `HDF5Reader` (local FS HDF5) | `h5py.File()` + dataset slice | `os.stat()` for byte count (HDF5 does not expose raw bytes cleanly) | Use file size from stat, skip h5py decode |
| `TFReader` (TFRecord) | Already returns `resized_image`, no decode | No change needed | ✅ Already correct |
| S3 iterable readers | Already raw byte read | No change needed | ✅ Already correct |

For `HDF5Reader`, full raw-byte skipping is complicated because HDF5 files contain many datasets and the per-sample byte cost is embedded inside the HDF5 container format. The pragmatic fix is to record the total file size (via `os.stat()`, which is already a real syscall) and use `ceil(file_size / num_samples_per_file)` as the per-sample byte count. This avoids `h5py` decoding while still exercising real storage I/O.

**Generator rationalization targets:**

The same data-loader-aware branch described in Section 9g applies to generation. For non-DALI data loaders, JPEG and PNG generators must write raw dgen-py bytes rather than running PIL encode. This produces files that the rationalized `ImageReader` reads correctly (raw bytes, byte count for telemetry). For the DALI path, PIL encode remains necessary and the DALI reader is already correct.

### 13e. Validation: How to Confirm the Fix Works

After rationalizing the local-FS readers, a correctly implemented benchmark should satisfy:

1. **A file-backend and object-backend run of the same workload with the same dataset produce statistically equivalent samples/sec and MiB/s numbers**, adjusted for storage latency and bandwidth differences between the two systems. CPU overhead should not be a confounding variable.

2. **The fraction of training-step time attributed to I/O wait (as reported in `dlp` traces) should be the dominant fraction (> 80%)** for both backends, for all formats, on any storage system faster than the benchmark's prefetch queue can drain.

3. **Generator throughput for JPEG/PNG on non-DALI configurations should match NPY generator throughput** (within 2×), because both should be bottlenecked on storage write bandwidth, not CPU encoding.

If any of these properties does not hold after rationalizing the readers, it indicates a remaining source of CPU overhead that has not been identified or removed.

### 13f. Configuration-Level Rationalization

Beyond code changes, the YAML configs should be audited to eliminate format choices that reflect historical defaults rather than deliberate workload simulation decisions:

1. **Local-FS configs that use JPEG/PNG for non-imaging workloads** (e.g., testing batch read throughput of random data) should be migrated to NPY or HDF5 with compression disabled. This eliminates generation overhead that is independent of the format rationalization.

2. **Object-store configs that use NPY/HDF5 while local-FS configs use JPEG/PNG for the "same" workload** create an implicit apples-to-oranges comparison. If a workload is defined as JPEG-format vision training, both its local-FS and object-store variants should use identical format settings. The storage backend is the variable; the format should be held constant.

3. **The `multiprocessing_context` coupling** (Section 6c) means that a rationalized file-backend config and its object-store counterpart must differ in at least one reader setting (`fork` vs `spawn`). This is unavoidable given the Tokio runtime constraint, but should be the ONLY difference between the two, and should be auto-derived from `storage_library` rather than manually set.

### 13g. Summary of the Rationalization Requirement

The core requirement is simple: **every reader, for every format, for every storage backend, must behave consistently.** The S3 iterable readers already implement the correct behaviour. The local-filesystem readers must be updated to match. Until that update is made, no published DLIO benchmark result comparing local-filesystem and object-storage throughput can be considered internally consistent, because the benchmarks are not measuring the same thing on both backends.
