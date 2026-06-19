"""
Object-storage integration tests — s3dlio + DLIOBenchmark
==========================================================

Verifies that every supported data format can be written (put) to and read
(get) from real S3-compatible object storage using the s3dlio library via the
standard DLIOBenchmark workflow:

  1. ``generate_data=True``  → DLIOBenchmark writes objects to the bucket.
  2. Verify object count in the bucket via ``s3dlio.list()``.
  3. ``train=True``          → DLIOBenchmark reads the objects back.

Storage configuration mirrors ``unet3d_h100_s3dlio_datagen.yaml``:
  storage_type: s3
  storage_library: s3dlio
  storage_root: <bucket>   (from DLIO_TEST_BUCKET, default: mlp-s3dlio)
  endpoint_url: from .env / AWS_ENDPOINT_URL

Opt-in gate
-----------
These tests hit a live MinIO endpoint and are NOT run by default.
Set the environment variable before running pytest::

    DLIO_S3_INTEGRATION=1 pytest tests/test_s3dlio_object_store.py -v

Credentials
-----------
Loaded from ``<repo>/.env``, with real environment variables taking priority
(same precedence as the shell scripts in tests/object-store/).

Formats tested
--------------
npy, npz, hdf5, csv, parquet, jpeg, png, tfrecord (full generate + read cycle).
All formats use s3dlio for both write and read.
"""

import os
import uuid
import logging
import shutil
import glob
from pathlib import Path

import pytest

# ─── Enable s3dlio / Rust-level tracing before any s3dlio import ──────────────
# RUST_LOG controls the log level of the Rust/Tokio layer inside s3dlio.
# Set to "info" so every PUT, GET, and LIST is visible in the test output.
# Override to "debug" for even more detail: RUST_LOG=debug pytest ...
os.environ.setdefault("RUST_LOG", "info")

# ─── Load credentials eagerly — must happen before s3dlio is imported ─────────
_REPO_ROOT = Path(__file__).parent.parent.parent   # mlp-storage/


def _load_env_file():
    """Load key=value pairs from .env, skipping keys already set by the shell."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            # Environment variable takes priority over .env file value.
            if key not in os.environ:
                os.environ[key] = val


_load_env_file()

# ─── Python-level logging — DEBUG so every benchmark step is traceable ────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,   # override any earlier basicConfig from conftest or dlio imports
)
# Keep noisy third-party loggers at INFO level.
for _noisy in ("urllib3", "s3transfer", "filelock", "hydra"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ─── Object-storage opt-in gate ──────────────────────────────────────────────
# These tests hit a live MinIO/S3 endpoint and are NOT run by default.
# Enable by setting the environment variable before running pytest:
#
#   DLIO_OBJECT_STORAGE_TESTS=1 pytest tests/test_s3dlio_object_store.py -v
#
# CI explicitly sets DLIO_OBJECT_STORAGE_TESTS=0, so these tests are always
# skipped during automated builds.
_OBJECT_TESTS_ENABLED = os.environ.get("DLIO_OBJECT_STORAGE_TESTS", "0") == "1"
if not _OBJECT_TESTS_ENABLED:
    pytest.skip(
        "Object-storage tests are disabled. Set DLIO_OBJECT_STORAGE_TESTS=1 to enable.",
        allow_module_level=True,
    )

# ─── DLIO test infrastructure ─────────────────────────────────────────────────
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mpi4py import MPI
import dlio_benchmark

from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from dlio_benchmark.main import DLIOBenchmark
# Per-test timeout.  4 train + 2 eval small objects (256-byte records) should
# generate and read back in well under 2 minutes on any reachable endpoint.
# Using a value much shorter than TEST_TIMEOUT_SECONDS (600 s) so a hang is
# caught quickly rather than after 10 minutes.
_S3_TEST_TIMEOUT = int(os.environ.get("DLIO_S3_TEST_TIMEOUT", "120"))  # seconds
# When DLIO_S3_EXTENDED=1, run all supported formats.  Default: npy only
# (fastest smoke test — verifies the put+get cycle without exhausting time).
_S3_EXTENDED = os.environ.get("DLIO_S3_EXTENDED", "0") == "1"

comm = MPI.COMM_WORLD
_config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"
_DLIO_TEST_OUTPUT_DIR = os.environ.get("DLIO_TEST_OUTPUT_DIR", "dlio_test_output")

log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _endpoint():
    ep = os.environ.get("AWS_ENDPOINT_URL")
    if not ep:
        pytest.skip("AWS_ENDPOINT_URL not set — cannot run live S3 tests")
    return ep


def _region():
    return os.environ.get("AWS_REGION", "us-east-1")


def _cleanup_s3_prefix(bucket: str, prefix: str) -> None:
    """
    Delete all objects under ``s3://bucket/prefix/`` using s3dlio.

    s3dlio.list() returns full URIs; s3dlio.delete() accepts a full URI.
    We list first then delete each object individually — the same pattern
    used in tests/object-store/dlio_s3dlio_cleanup.sh.
    """
    import s3dlio
    # Ensure trailing slash so listing is prefix-scoped.
    list_uri = f"s3://{bucket}/{prefix.lstrip('/')}".rstrip('/') + '/'
    log.info("cleanup: listing %s ...", list_uri)
    try:
        uris = s3dlio.list(list_uri, recursive=True)
    except Exception as exc:
        log.warning("cleanup: s3dlio.list(%r) raised: %s", list_uri, exc)
        return
    log.info("cleanup: deleting %d object(s) under %s", len(uris), list_uri)
    for uri in uris:
        log.debug("cleanup: delete %s", uri)
        try:
            s3dlio.delete(uri)
        except Exception as exc:
            log.warning("cleanup: s3dlio.delete(%r) raised: %s", uri, exc)
    log.info("cleanup: done — deleted %d object(s)", len(uris))


def _list_objects_s3dlio(uri: str) -> list:
    """List objects under a URI using s3dlio (returns full URIs)."""
    import s3dlio
    log.debug("list: s3dlio.list(%r, recursive=True)", uri)
    try:
        result = s3dlio.list(uri, recursive=True)
        log.debug("list: found %d object(s)", len(result))
        return result
    except Exception as exc:
        log.warning("s3dlio.list(%r) raised: %s", uri, exc)
        return []


def _run_benchmark(workload_dict: dict, phase: str = "", verify: bool = False) -> DLIOBenchmark:
    """Instantiate and run DLIOBenchmark, returning the benchmark object."""
    tag = f"[{phase}] " if phase else ""
    log.info("%sDLIOBenchmark starting — workflow=%s",
             tag, workload_dict.get("workflow", {}))
    comm.Barrier()
    ConfigArguments.reset()
    workload_dict.setdefault("output", {})["folder"] = _DLIO_TEST_OUTPUT_DIR
    bench = DLIOBenchmark(workload_dict)
    log.info("%sinitialize ...", tag)
    bench.initialize()
    log.info("%srun ...", tag)
    bench.run()
    log.info("%sfinalize ...", tag)
    bench.finalize()
    comm.Barrier()
    log.info("%sDLIOBenchmark complete", tag)
    if comm.rank == 0 and verify:
        output_pattern = os.path.join(bench.output_folder, "*_output.json")
        output_jsons = glob.glob(output_pattern)
        assert len(output_jsons) == bench.comm_size, (
            f"Expected {bench.comm_size} output JSON(s), found {len(output_jsons)}"
        )
    return bench


# ─── Base Hydra overrides shared across all format tests ──────────────────────

def _base_overrides(bucket: str, prefix: str, fmt: str,
                    num_train: int = 4, num_eval: int = 2) -> list:
    """
    Build the common Hydra overrides for s3dlio object-storage tests.

    Maps directly to the storage: section in unet3d_h100_s3dlio_datagen.yaml:
      storage_type: s3
      storage_root: <bucket>        ← namespace / bucket name
      storage_library: s3dlio       ← promoted into storage_options by config.py
      storage_options.endpoint_url  ← custom MinIO / VAST endpoint
      storage_options.region        ← AWS region (us-east-1)

    dataset.data_folder is a path relative to the bucket; DLIO appends /train/
    and /valid/ automatically.  Object URIs become:
      s3://<bucket>/<prefix>/train/img_train_XXXXXXXX.<fmt>
    """
    return [
        # Framework: pytorch → StorageFactory dispatches to ObjStoreLibStorage → s3dlio
        "++workload.framework=pytorch",
        "++workload.reader.data_loader=pytorch",

        # Storage: real s3dlio against live MinIO endpoint
        "++workload.storage.storage_type=s3",
        f"++workload.storage.storage_root={bucket}",
        "++workload.storage.storage_library=s3dlio",
        f"++workload.storage.storage_options.endpoint_url={_endpoint()}",
        f"++workload.storage.storage_options.region={_region()}",

        # Dataset: small files for quick verification
        f"++workload.dataset.data_folder={prefix}",
        f"++workload.dataset.format={fmt}",
        f"++workload.dataset.num_files_train={num_train}",
        f"++workload.dataset.num_files_eval={num_eval}",
        "++workload.dataset.num_samples_per_file=4",
        "++workload.dataset.record_length=256",    # 256 bytes → 16×16 for images
        "++workload.dataset.record_length_stdev=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Integration test: datagen (put) + list (verify) + train (get) for each format
# ═══════════════════════════════════════════════════════════════════════════════

_FORMATS = ["npy"] if not _S3_EXTENDED else ["npy", "npz", "hdf5", "csv", "parquet", "jpeg", "png"]
# TFRecord excluded: reading requires framework=tensorflow which routes through
# S3Storage (tf.io.gfile), not ObjStoreLibStorage (s3dlio).  Generate-only test
# (TFRecord full datagen+read tested separately in test_s3dlio_tfrecord_datagen_and_read).
# for TFRecord is covered by test_s3dlio_tfrecord_datagen below.


@pytest.mark.timeout(_S3_TEST_TIMEOUT, method="thread")
@pytest.mark.parametrize("fmt", _FORMATS)
def test_s3dlio_datagen_and_read(fmt):
    """
    Full put+get cycle for *fmt* via DLIOBenchmark + s3dlio.

    Phase 1 — generate_data=True:
      DLIOBenchmark calls the format-specific generator, serialises the data,
      and writes each object via ObjStoreLibStorage → s3dlio → MinIO.

    Phase 2 — verify object count:
      s3dlio.list() confirms the expected number of objects are visible in the
      bucket under the test prefix.

    Phase 3 — train=True:
      DLIOBenchmark reads every object back using the matching DLIO reader
      (e.g. NpzReader for npz) via s3dlio.get().
    """
    DLIOMPI.get_instance().initialize()

    bucket  = os.environ.get("DLIO_TEST_BUCKET", "mlp-s3dlio")
    run_id  = str(uuid.uuid4())[:8]
    prefix  = f"dlio-pytest/{run_id}/{fmt}"

    num_train = 4
    num_eval  = 2

    log.info(
        "test_s3dlio_datagen_and_read[%s]: bucket=%s prefix=%s endpoint=%s RUST_LOG=%s",
        fmt, bucket, prefix, _endpoint(), os.environ.get("RUST_LOG", "(unset)"),
    )

    base = _base_overrides(bucket, prefix, fmt, num_train=num_train, num_eval=num_eval)

    try:
        with initialize_config_dir(version_base=None, config_dir=_config_dir):

            # ── Phase 1: write objects ────────────────────────────────────────
            log.info("[%s] Phase 1: generate_data → writing %d train + %d eval objects",
                     fmt, num_train, num_eval)
            cfg = compose(config_name="config", overrides=base + [
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.workflow.checkpoint=False",
            ])
            _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                           phase="datagen", verify=False)

            # ── Phase 2: verify objects in bucket ────────────────────────────
            train_uri = f"s3://{bucket}/{prefix}/train/"
            valid_uri = f"s3://{bucket}/{prefix}/valid/"

            found_train = _list_objects_s3dlio(train_uri)
            found_valid = _list_objects_s3dlio(valid_uri)

            log.info("[%s] Phase 2: found %d train, %d valid objects",
                     fmt, len(found_train), len(found_valid))

            assert len(found_train) == num_train, (
                f"[{fmt}] Expected {num_train} train objects at {train_uri}, "
                f"found {len(found_train)}: {found_train}"
            )
            assert len(found_valid) == num_eval, (
                f"[{fmt}] Expected {num_eval} valid objects at {valid_uri}, "
                f"found {len(found_valid)}: {found_valid}"
            )

            # ── Phase 3: read objects back ────────────────────────────────────
            log.info("[%s] Phase 3: train → reading objects back", fmt)
            ConfigArguments.reset()
            cfg = compose(config_name="config", overrides=base + [
                "++workload.workflow.generate_data=False",
                "++workload.workflow.train=True",
                "++workload.workflow.checkpoint=False",
                "++workload.train.epochs=1",
                "++workload.train.computation_time=0.0",
                "++workload.reader.read_threads=0",  # 0 = main thread, avoids fork() deadlock under pytest
                "++workload.reader.batch_size=2",
            ])
            _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                           phase="train", verify=True)

            log.info("[%s] PASSED — put and get both succeeded", fmt)

    finally:
        # Always clean up test objects so we don't pollute the bucket.
        if comm.rank == 0:
            _cleanup_s3_prefix(bucket, f"{prefix}/")
        # Clean up any local DLIO output files.
        shutil.rmtree(_DLIO_TEST_OUTPUT_DIR, ignore_errors=True)


# ─── TFRecord: full generate + read test ─────────────────────────────────────

@pytest.mark.timeout(_S3_TEST_TIMEOUT, method="thread")
def test_s3dlio_tfrecord_datagen_and_read():
    """
    Full generate + read test for TFRecord format using s3dlio.

    TFRecord generation writes objects via s3dlio (TFRecordGenerator + put_data).
    Reading uses TFRecordReaderS3Iterable which fetches raw bytes via s3dlio
    get_many() — no tensorflow/protobuf decoding required.  Both phases use
    framework=pytorch so no tensorflow installation is needed.
    """
    DLIOMPI.get_instance().initialize()

    bucket = os.environ.get("DLIO_TEST_BUCKET", "mlp-s3dlio")
    run_id = str(uuid.uuid4())[:8]
    prefix = f"dlio-pytest/{run_id}/tfrecord"

    num_train = 4
    num_eval  = 2

    log.info("test_s3dlio_tfrecord_datagen: bucket=%s prefix=%s endpoint=%s RUST_LOG=%s",
             bucket, prefix, _endpoint(), os.environ.get("RUST_LOG", "(unset)"))

    base = _base_overrides(bucket, prefix, "tfrecord",
                           num_train=num_train, num_eval=num_eval)

    try:
        with initialize_config_dir(version_base=None, config_dir=_config_dir):
            cfg = compose(config_name="config", overrides=base + [
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.checkpoint=False",
            ])
            _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                           phase="datagen", verify=False)

        train_uri = f"s3://{bucket}/{prefix}/train/"
        valid_uri = f"s3://{bucket}/{prefix}/valid/"

        found_train = _list_objects_s3dlio(train_uri)
        found_valid = _list_objects_s3dlio(valid_uri)

        log.info("tfrecord: found %d train, %d valid objects",
                 len(found_train), len(found_valid))

        assert len(found_train) == num_train, (
            f"[tfrecord] Expected {num_train} train objects at {train_uri}, "
            f"found {len(found_train)}: {found_train}"
        )
        assert len(found_valid) == num_eval, (
            f"[tfrecord] Expected {num_eval} valid objects at {valid_uri}, "
            f"found {len(found_valid)}: {found_valid}"
        )

        log.info("tfrecord datagen PASSED — now running read phase ...")

        # Read phase: TFRecordReaderS3Iterable fetches objects via s3dlio.
        # Uses files_pre_sharded=True so DLIO does not attempt to re-list S3.
        with initialize_config_dir(version_base=None, config_dir=_config_dir):
            cfg = compose(config_name="config", overrides=base + [
                "++workload.workflow.generate_data=False",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.checkpoint=False",
                "++workload.dataset.files_pre_sharded=True",
                "++workload.train.epochs=1",
            ])
            _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                           phase="train", verify=True)

        log.info("test_s3dlio_tfrecord_datagen_and_read PASSED — put + get confirmed")

    finally:
        if comm.rank == 0:
            _cleanup_s3_prefix(bucket, f"{prefix}/")
        shutil.rmtree(_DLIO_TEST_OUTPUT_DIR, ignore_errors=True)
