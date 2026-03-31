"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

PyTorchObjStoreCheckpointing — streaming checkpoint for minio / s3dlio / s3torchconnector
------------------------------------------------------------------------------------------
Unlike PyTorchCheckpointing (which allocates full model tensors in RAM and then
serialises them with torch.save), this class uses the
mlpstorage.checkpointing.StreamingCheckpointing producer-consumer pipeline:

  • get_tensor_core() returns a _SizePlaceholder instead of a real torch
    tensor.  No model tensors are ever allocated in RAM, so even 70B+
    parameter models fit in a few hundred MiB per MPI process.

  • save_state() sums the per-placeholder byte counts, then calls
    StreamingCheckpointing.save(uri, total_bytes).  dgen-py generates
    synthetic random data of the same byte count while the storage library
    (minio, s3dlio, or s3torchconnector) streams it to the object store.  Peak RAM ≈ 128 MiB (4 × 32 MiB
    buffer pool).

  • load_state() computes the expected byte count from the same placeholders
    and calls StreamingCheckpointing.load(uri, total_bytes) which executes
    parallel range-GETs and discards received data immediately.  The
    in-memory state dict is left unchanged so callers' assertions pass.

Credential propagation
~~~~~~~~~~~~~~~~~~~~~~
Credentials are written to the standard environment variables
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3 /
AWS_ENDPOINT_URL) during __init__, so the forked writer process inherits
them automatically.

storage_type: s3  +  storage_library: minio            →  this class
storage_type: s3  +  storage_library: s3dlio           →  this class
storage_type: s3  +  storage_library: s3torchconnector →  this class
"""

import os

from dlio_benchmark.checkpointing.base_checkpointing import (
    BaseCheckpointing,
    get_datatype_size,
)
from dlio_benchmark.checkpointing.pytorch_checkpointing import (
    PyTorchCheckpointing,
    _SizePlaceholder,
    _compute_state_bytes,
)
from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from dlio_benchmark.utils.utility import Profile, dft_ai

dlp = Profile(MODULE_CHECKPOINT)


# _SizePlaceholder and _compute_state_bytes are defined in pytorch_checkpointing
# and imported above to avoid duplication.  See that module for documentation.

class PyTorchObjStoreCheckpointing(PyTorchCheckpointing):
    """Streaming checkpoint backed by minio, s3dlio, or s3torchconnector.

    get_tensor_core() is overridden to return _SizePlaceholder objects so no
    large tensor allocations occur during __init__.  save_state() and
    load_state() use mlpstorage.checkpointing.StreamingCheckpointing which
    runs a dgen-py producer and a storage-backend consumer in a
    producer-consumer pipeline.  Peak RAM is fixed at ~128 MiB (4 × 32 MiB
    buffer pool) regardless of checkpoint size.

    storage_type: s3  +  storage_library: minio            →  this class
    storage_type: s3  +  storage_library: s3dlio           →  this class
    storage_type: s3  +  storage_library: s3torchconnector →  this class
    """

    __instance = None

    @staticmethod
    def get_instance():
        if PyTorchObjStoreCheckpointing.__instance is None:
            PyTorchObjStoreCheckpointing.__instance = PyTorchObjStoreCheckpointing()
        return PyTorchObjStoreCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        # BaseCheckpointing.__init__ calls self.get_tensor_core() to build the
        # state dicts.  Our override below returns _SizePlaceholder objects so
        # nothing large is allocated here.
        BaseCheckpointing.__init__(self, "pt")

        storage_options = getattr(self.args, "storage_options", {}) or {}
        # storage_library is REQUIRED — there is no default.  Every object storage
        # workload must explicitly declare which library to use via
        # storage_options["storage_library"] (set by storage_library: in the YAML
        # or via storage.storage_options.storage_library=<value> on the CLI).
        self.storage_library = storage_options.get("storage_library")
        if self.storage_library is None:
            raise ValueError(
                "storage_options['storage_library'] is required for "
                "PyTorchObjStoreCheckpointing. Add 'storage_library: <value>' "
                "under the 'storage:' section of your workload YAML. "
                "Supported values: minio, s3dlio, s3torchconnector."
            )
        self.access_key_id    = storage_options.get("access_key_id")
        self.secret_access_key = storage_options.get("secret_access_key")
        self.endpoint         = storage_options.get("endpoint_url")
        self.region           = storage_options.get(
            "region", getattr(self.args, "s3_region", "us-east-1")
        )

        # Write credentials to env — the forked writer process inherits them.
        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        if self.storage_library == "s3dlio":
            if self.endpoint:
                os.environ["AWS_ENDPOINT_URL_S3"] = self.endpoint
            try:
                import s3dlio  # noqa: F401  validates installation
            except ImportError as exc:
                raise ImportError(
                    "storage_library=s3dlio is configured but s3dlio is not "
                    "installed.  Install with: pip install s3dlio"
                ) from exc

        elif self.storage_library == "minio":
            if self.endpoint:
                os.environ["AWS_ENDPOINT_URL"] = self.endpoint

        elif self.storage_library == "s3torchconnector":
            if self.endpoint:
                os.environ["AWS_ENDPOINT_URL"] = self.endpoint
            try:
                import s3torchconnector  # noqa: F401  validates installation
            except ImportError as exc:
                raise ImportError(
                    "storage_library=s3torchconnector is configured but "
                    "s3torchconnector is not installed. "
                    "Install with: pip install s3torchconnectorclient"
                ) from exc

        else:
            raise ValueError(
                f"PyTorchObjStoreCheckpointing does not support "
                f"storage_library='{self.storage_library}'. "
                f"Use 'minio', 's3dlio', or 's3torchconnector'."
            )

        # Build StreamingCheckpointing once; reused for all save/load calls.
        try:
            from mlpstorage.checkpointing import StreamingCheckpointing as _SC
        except ImportError as exc:
            raise ImportError(
                "Object-store checkpointing requires mlpstorage. "
                "Install mlpstorage in this environment to use "
                "storage_library=minio/s3dlio/s3torchconnector checkpointing."
            ) from exc

        # Detect MPI world size to throttle per-rank concurrency.
        # With 8 MPI ranks each uploading concurrently, per-rank parallelism
        # must be reduced to avoid overwhelming the storage target.
        _mpi_world_size = 1
        for _env in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
            _ev = os.environ.get(_env)
            if _ev:
                try:
                    _mpi_world_size = max(1, int(_ev))
                    break
                except ValueError:
                    pass

        streaming_kwargs: dict = dict(
            chunk_size=32 * 1024 * 1024,   # 32 MiB write chunks
            num_buffers=4,                  # 4 × 32 MiB = 128 MiB pool
            use_dgen=True,
            backend=self.storage_library,
            num_parallel_readers=max(2, 8 // _mpi_world_size),
        )
        if self.storage_library == "minio":
            # Throttle minio thread pool proportionally to MPI world size
            streaming_kwargs.update(
                part_size=32 * 1024 * 1024,
                num_parallel_uploads=max(2, 8 // _mpi_world_size),
            )
        elif self.storage_library == "s3dlio":
            # s3dlio multipart upload tuning.
            #
            # Background (v0.9.82 regression):
            #   spawn_part() acquires the concurrency semaphore *before* spawning the
            #   upload task, blocking the Python writer thread until a slot is free.
            #   This prevents an OOM/runtime-overload bug (pre-v0.9.82 code spawned all
            #   parts simultaneously — ~467 tasks × 32 MiB = ~15 GiB Rust heap for a
            #   14.96 GiB object) but at the cost of pipeline stalls.
            #
            # Tuning levers:
            #   S3DLIO_MULTIPART_PART_SIZE_MiB  — part size in MiB (default: 16)
            #       Larger parts → fewer semaphore trips but each stall lasts longer.
            #       Smaller parts + more max_in_flight → more concurrent MinIO connections.
            #   S3DLIO_MULTIPART_MAX_IN_FLIGHT — concurrent upload slots (default: 16)
            #       More slots → more parallel MinIO UploadPart connections per object.
            #       Peak Rust memory = max_in_flight × part_size_bytes.
            #
            # Benchmark matrix (env-var driven — no code change needed):
            #   16 MiB × 16 slots  →  256 MiB  peak, 16 connections  (library default)
            #   16 MiB × 32 slots  →  512 MiB  peak, 32 connections
            #   16 MiB × 64 slots  →    1 GiB peak, 64 connections
            #   32 MiB × 32 slots  →    1 GiB peak, 32 connections
            #   64 MiB × 16 slots  →    1 GiB peak, 16 connections
            #  128 MiB ×  8 slots  →    1 GiB peak,  8 connections  (previous default)
            #
            # Root-cause fix (tracked in GitHub issue #134):
            #   A coordinator Tokio task + bounded mpsc::channel will make the Python
            #   writer non-blocking regardless of max_in_flight, eliminating stalls.
            #   Until that fix lands, maximising max_in_flight within memory budget
            #   is the best available workaround.
            #
            _part_size_mib   = int(os.environ.get("S3DLIO_MULTIPART_PART_SIZE_MiB",  "128"))
            _max_in_flight  = int(os.environ.get("S3DLIO_MULTIPART_MAX_IN_FLIGHT", "8"))
            streaming_kwargs.update(
                part_size=_part_size_mib * 1024 * 1024,
                max_in_flight=_max_in_flight,
            )

        self._streaming = _SC(**streaming_kwargs)

    # ------------------------------------------------------------------
    # Override get_tensor_core — return placeholder, not a real tensor
    # ------------------------------------------------------------------

    def get_tensor_core(self, length, datatype="int8", randomize=True):
        """Return a _SizePlaceholder that records byte-size without allocating."""
        return _SizePlaceholder(length, datatype)

    # ------------------------------------------------------------------
    # save_state / load_state
    # ------------------------------------------------------------------

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync=False):
        """Stream synthetic data of the correct byte-count to object storage.

        The byte count is derived from _SizePlaceholder values in *state* and
        matches what torch.save(state) would produce.  fsync is ignored —
        object storage does not expose fsync semantics.
        """
        uri         = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)

        if total_bytes <= 0:
            self.logger.warning(
                f"save_state: computed 0 bytes for suffix '{suffix}', skipping"
            )
            return

        self._streaming.save(uri, total_bytes)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        """Stream-read the checkpoint from object storage and discard data.

        The in-memory *state* dict (holding _SizePlaceholder objects) is left
        unchanged so that callers' `assert len(state.keys()) > 0` continues
        to pass — this is a throughput benchmark, not a training restore.
        """
        uri         = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)

        if total_bytes <= 0:
            self.logger.warning(
                f"load_state: computed 0 bytes for suffix '{suffix}', skipping"
            )
            return

        self._streaming.load(uri, total_bytes)
        assert len(state.keys()) > 0

    # ------------------------------------------------------------------
    # Delegate orchestration hooks up the MRO
    # ------------------------------------------------------------------

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()
