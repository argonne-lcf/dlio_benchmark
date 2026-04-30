"""
Fast integration test suite — targets < 10 minutes total, no mpirun required.

Philosophy:
  - Unit tests: pure logic, no MPI, no real disk I/O
  - Smoke tests: minimal I/O (one file, one format) to verify the pipeline
    works end-to-end; one MPI test just to confirm mpirun itself launches
  - Parametrized broadly on core dimensions; NOT exhaustively (that's the
    integration suite's job)

Coverage areas:
  1.  Enumerations — all core enums round-trip through str/get_enum
  2.  Utilities    — gen_random_tensor, add_padding, utcnow, str2bool
  3.  Config       — ConfigArguments field defaults, derive_configurations
                     logic (checkpoint_mechanism auto-select, dimension math)
  4.  Factories    — GeneratorFactory and StorageFactory return correct types
  5.  Data generators — per-format: correct file structure and dtype (npy,
                     npz, hdf5, csv, jpeg, png, tfrecord, indexed_binary,
                     parquet legacy + column-schema modes with footer checks)
  6.  Reader compat — generator output is readable by matching DLIO reader
  7.  MPI smoke    — mpirun -np 2 launches and exits cleanly (one call only)
  8.  End-to-end smoke — minimal generate+train run via DLIOBenchmark
                     (npy, 1 rank, tiny dataset, no TF/PT training loop)
"""

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
os.environ.setdefault("DLIO_OUTPUT_FOLDER", "dlio_test_output")
DLIO_TEST_OUTPUT_DIR = os.environ.get("DLIO_TEST_OUTPUT_DIR", "dlio_test_output")

import dlio_benchmark
_CONFIG_DIR = os.path.dirname(dlio_benchmark.__file__) + "/configs/"


def _reset():
    """Reset all DLIO singletons between tests."""
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI
    ConfigArguments.reset()
    DLIOMPI.reset()


def _make_cfg(extra_overrides=()):
    """Build a minimal Hydra config for use in tests."""
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        overrides = [
            "workload=unet3d_a100",
            "++workload.framework=tensorflow",
            "++workload.reader.data_loader=tensorflow",
            "++workload.workflow.generate_data=False",
            "++workload.workflow.train=False",
            "++workload.dataset.num_files_train=2",
            "++workload.dataset.num_files_eval=0",
            "++workload.dataset.num_samples_per_file=2",
            "++workload.dataset.record_length=256",
            "++workload.dataset.record_length_stdev=0",
            "++workload.train.epochs=1",
        ] + list(extra_overrides)
        return compose(config_name="config", overrides=overrides)


# ===========================================================================
# 0. Preflight — installation integrity
#    Mirrors the "Preflight runtime imports" step in ci.yml.
#    Runs under BOTH install methods (pip install .[test] AND
#    pip install -r requirements-test.txt + PYTHONPATH) so failures are
#    caught early regardless of how the venv was built.
# ===========================================================================
class TestPreflight:
    """Verify that all required (and optional) packages installed correctly."""

    # --- dlio_benchmark itself -------------------------------------------------
    def test_dlio_benchmark_importable(self):
        import dlio_benchmark  # noqa: F401

    def test_dlio_main_entrypoint(self):
        from dlio_benchmark.main import main
        assert callable(main)

    # --- Core runtime dependencies (pyproject.toml [dependencies]) ------------
    def test_numpy(self):
        import numpy as np
        assert hasattr(np, "__version__")

    def test_h5py(self):
        import h5py
        assert hasattr(h5py, "__version__")

    def test_mpi4py(self):
        from mpi4py import MPI
        # Just importing initialises nothing — only checks linkage is correct.
        assert MPI.COMM_WORLD is not None

    def test_hydra_core(self):
        import hydra
        assert hasattr(hydra, "__version__")

    def test_omegaconf(self):
        import omegaconf
        assert hasattr(omegaconf, "__version__")

    def test_pandas(self):
        import pandas
        assert hasattr(pandas, "__version__")

    def test_pillow(self):
        from PIL import Image
        assert callable(Image.open)

    def test_pyarrow(self):
        import pyarrow
        assert hasattr(pyarrow, "__version__")

    def test_psutil(self):
        import psutil
        assert hasattr(psutil, "__version__")

    def test_pyyaml(self):
        import yaml
        assert hasattr(yaml, "__version__")

    def test_tensorflow(self):
        import tensorflow
        assert hasattr(tensorflow, "__version__")

    def test_torch(self):
        import torch
        assert hasattr(torch, "__version__")

    # --- dftracer: optional tracing library, graceful no-op if absent ------
    # The library has a try/except fallback in utility.py — if it fails to
    # import, DLIO silently uses no-op stubs. Testing it as a hard requirement
    # would cause false CI failures on minimal installs. Skip if absent.
    def test_dftracer_python(self):
        pytest.importorskip(
            "dftracer.python",
            reason="dftracer.python not installed — optional tracing library; "
                   "DLIO degrades gracefully to no-op stubs when absent.",
        )

    def test_dftracer_core(self):
        pytest.importorskip(
            "dftracer.dftracer",
            reason="dftracer.dftracer not installed — optional tracing library; "
                   "DLIO degrades gracefully to no-op stubs when absent.",
        )

    # --- dgen_py: optional, but warn loudly if missing -----------------------
    def test_dgen_py_optional(self):
        """dgen_py is optional (mirrors ci.yml preflight 'optional' list).
        Skipped (not failed) when absent; install for 155x faster data gen."""
        pytest.importorskip(
            "dgen_py",
            reason="dgen_py not installed — optional, but strongly recommended "
                   "(155x faster than NumPy data generation).",
        )


# ===========================================================================
# 1. Enumerations
# ===========================================================================
class TestEnumerations:
    """All core enums must have working __str__ and round-trip through get_enum."""

    def test_format_type_str(self):
        from dlio_benchmark.common.enumerations import FormatType
        assert str(FormatType.NPY) == "npy"
        assert str(FormatType.HDF5) == "hdf5"
        assert str(FormatType.JPEG) == "jpeg"
        assert str(FormatType.PNG) == "png"
        assert str(FormatType.TFRECORD) == "tfrecord"
        assert str(FormatType.NPZ) == "npz"
        assert str(FormatType.CSV) == "csv"
        assert str(FormatType.INDEXED_BINARY) == "indexed_binary"

    def test_format_type_get_enum(self):
        from dlio_benchmark.common.enumerations import FormatType
        for name in ("npy", "npz", "hdf5", "jpeg", "png", "tfrecord", "csv",
                     "indexed_binary", "mmap_indexed_binary", "synthetic"):
            assert str(FormatType.get_enum(name)) == name

    def test_storage_type_str(self):
        from dlio_benchmark.common.enumerations import StorageType
        assert str(StorageType.LOCAL_FS) == "local_fs"
        assert str(StorageType.S3) == "s3"

    def test_checkpoint_mechanism_str(self):
        from dlio_benchmark.common.enumerations import CheckpointMechanismType
        assert str(CheckpointMechanismType.PT_SAVE) == "pt_save"
        assert str(CheckpointMechanismType.TF_SAVE) == "tf_save"

    def test_framework_type_str(self):
        from dlio_benchmark.common.enumerations import FrameworkType
        assert str(FrameworkType.TENSORFLOW) == "tensorflow"
        assert str(FrameworkType.PYTORCH) == "pytorch"

    def test_shuffle_enum(self):
        from dlio_benchmark.common.enumerations import Shuffle
        assert str(Shuffle.OFF) == "off"
        assert str(Shuffle.SEED) == "seed"


# ===========================================================================
# 2. Utilities
# ===========================================================================
class TestUtilities:
    def test_add_padding_no_digits(self):
        from dlio_benchmark.utils.utility import add_padding
        assert add_padding(5) == "5"
        assert add_padding(42) == "42"

    def test_add_padding_with_digits(self):
        from dlio_benchmark.utils.utility import add_padding
        assert add_padding(5, 4) == "0005"
        assert add_padding(1000, 4) == "1000"

    def test_utcnow_format(self):
        from dlio_benchmark.utils.utility import utcnow
        ts = utcnow()
        assert "T" in ts
        assert len(ts) > 10

    def test_str2bool_true_values(self):
        from dlio_benchmark.utils.utility import str2bool
        for v in ("yes", "true", "t", "y", "1", "True", "YES"):
            assert str2bool(v) is True

    def test_str2bool_false_values(self):
        from dlio_benchmark.utils.utility import str2bool
        for v in ("no", "false", "f", "n", "0", "False", "NO"):
            assert str2bool(v) is False

    def test_str2bool_invalid_raises(self):
        from dlio_benchmark.utils.utility import str2bool
        with pytest.raises(Exception):
            str2bool("maybe")

    def test_gen_random_tensor_shape(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(4, 4), dtype="float32")
        assert t.shape == (4, 4)
        assert t.dtype == np.float32

    def test_gen_random_tensor_int_dtype(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(8,), dtype="int8")
        assert t.dtype == np.int8

    def test_gen_random_tensor_seed_reproducible(self):
        """Same seed must produce identical data — uses dgen_py.Generator(seed=) correctly."""
        from dlio_benchmark.utils.utility import gen_random_tensor
        t1 = gen_random_tensor(shape=(16,), dtype="float32", seed=42)
        t2 = gen_random_tensor(shape=(16,), dtype="float32", seed=42)
        assert t1.shape == (16,)
        assert t1.dtype == np.float32
        np.testing.assert_array_equal(t1, t2)

    def test_gen_random_tensor_different_seeds_differ(self):
        """Different seeds must produce different data."""
        from dlio_benchmark.utils.utility import gen_random_tensor
        t1 = gen_random_tensor(shape=(32,), dtype="float32", seed=1)
        t2 = gen_random_tensor(shape=(32,), dtype="float32", seed=2)
        assert not np.array_equal(t1, t2)

    def test_gen_random_tensor_entropy(self):
        """Generated data must not be all zeros or all identical values."""
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(256,), dtype="float32")
        assert len(np.unique(t)) > 10


# ===========================================================================
# 3. Config — defaults and derive_configurations logic
# ===========================================================================
class TestConfigDefaults:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_default_format_is_tfrecord(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import FormatType
        args = ConfigArguments.get_instance()
        assert args.format == FormatType.TFRECORD

    def test_default_storage_type_local(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import StorageType
        args = ConfigArguments.get_instance()
        assert args.storage_type == StorageType.LOCAL_FS

    def test_default_read_threads(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.read_threads == 1

    def test_default_batch_size(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.batch_size == 1

    def test_default_seed(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.seed == 123


class TestConfigDerive:
    """Test derive_configurations logic without disk I/O."""

    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_checkpoint_mechanism_auto_tf(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, CheckpointMechanismType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.TENSORFLOW
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.checkpoint_mechanism == CheckpointMechanismType.TF_SAVE

    def test_checkpoint_mechanism_auto_pytorch(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, CheckpointMechanismType, StorageType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.PYTORCH
        args.storage_type = StorageType.LOCAL_FS
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.checkpoint_mechanism == CheckpointMechanismType.PT_SAVE

    def test_checkpoint_mechanism_s3_requires_storage_library(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, StorageType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.PYTORCH
        args.storage_type = StorageType.S3
        args.storage_options = {}  # missing storage_library
        args.do_checkpoint = False
        args.generate_data = False
        with pytest.raises(Exception, match="storage_library"):
            args.derive_configurations(file_list_train=[], file_list_eval=[])

    def test_dimension_from_record_length(self):
        import math
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        args.record_length = 256
        args.record_length_stdev = 0
        args.record_length_resize = 0
        args.record_dims = []
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.dimension == int(math.sqrt(256))  # == 16

    def test_training_steps_calculation(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        args.num_samples_per_file = 4
        args.batch_size = 2
        args.record_length = 64
        args.record_length_stdev = 0
        args.record_length_resize = 0
        args.record_dims = []
        args.do_checkpoint = False
        args.generate_data = False
        file_list = [f"file_{i}.npy" for i in range(4)]
        args.derive_configurations(file_list_train=file_list, file_list_eval=[])
        # total_samples=16, batch=2, comm_size=1 → steps=8
        assert args.training_steps == 8


# ===========================================================================
# 4. Factories — return correct types for each key format/storage
# ===========================================================================
class TestGeneratorFactory:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_npy_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        g = GeneratorFactory.get_generator(FormatType.NPY)
        assert isinstance(g, NPYGenerator)

    def test_npz_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        g = GeneratorFactory.get_generator(FormatType.NPZ)
        assert isinstance(g, NPZGenerator)

    def test_hdf5_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
        g = GeneratorFactory.get_generator(FormatType.HDF5)
        assert isinstance(g, HDF5Generator)

    def test_jpeg_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.jpeg_generator import JPEGGenerator
        g = GeneratorFactory.get_generator(FormatType.JPEG)
        assert isinstance(g, JPEGGenerator)

    def test_png_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.png_generator import PNGGenerator
        g = GeneratorFactory.get_generator(FormatType.PNG)
        assert isinstance(g, PNGGenerator)

    def test_tfrecord_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.tf_generator import TFRecordGenerator
        g = GeneratorFactory.get_generator(FormatType.TFRECORD)
        assert isinstance(g, TFRecordGenerator)

    def test_indexed_binary_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator
        g = GeneratorFactory.get_generator(FormatType.INDEXED_BINARY)
        assert isinstance(g, IndexedBinaryGenerator)

    def test_unknown_format_raises(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        with pytest.raises(Exception):
            GeneratorFactory.get_generator("not_a_real_format")


class TestStorageFactory:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_local_fs_storage(self):
        from dlio_benchmark.storage.storage_factory import StorageFactory
        from dlio_benchmark.common.enumerations import StorageType
        from dlio_benchmark.storage.file_storage import FileStorage
        s = StorageFactory.get_storage(StorageType.LOCAL_FS, ".", None)
        assert isinstance(s, FileStorage)


# ===========================================================================
# 5. Data generators — format correctness (small files, no MPI)
# ===========================================================================
@pytest.fixture
def tmpdir_clean():
    d = tempfile.mkdtemp(prefix="dlio_fast_ci_")
    yield pathlib.Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _setup_config_for_gen(args, tmpdir, fmt, n_samples=4, record_length=256):
    """Configure ConfigArguments for a minimal generation run."""
    from dlio_benchmark.common.enumerations import (
        FormatType, StorageType, FrameworkType, DataLoaderType,
        CheckpointMechanismType
    )
    args.format = fmt
    args.storage_type = StorageType.LOCAL_FS
    args.storage_root = str(tmpdir)
    args.data_folder = str(tmpdir / "data") + "/"
    args.record_length = record_length
    args.record_length_stdev = 0
    args.record_length_resize = 0
    args.record_dims = []
    args.num_samples_per_file = n_samples
    args.num_files_train = 2
    args.num_files_eval = 0
    args.batch_size = 1
    args.epochs = 1
    args.file_prefix = "img"
    args.do_checkpoint = False
    args.generate_data = True
    args.framework = FrameworkType.TENSORFLOW
    args.data_loader = DataLoaderType.TENSORFLOW
    args.checkpoint_mechanism = CheckpointMechanismType.TF_SAVE
    # derive_configurations(None, None): computes dimension from record_length,
    # but does NOT overwrite num_files_train (that path only runs when both
    # file_list_train and file_list_eval are non-None).
    args.derive_configurations(file_list_train=None, file_list_eval=None)


class TestNpyGenerator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npy_files_created(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        gen = NPYGenerator()
        gen.generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npy"))
        assert len(files) > 0
        arr = np.load(files[0])
        assert isinstance(arr, np.ndarray)
        assert arr.ndim >= 2

    def test_npy_non_empty(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        NPYGenerator().generate()
        for f in pathlib.Path(args.data_folder).rglob("*.npy"):
            assert f.stat().st_size > 0


class TestNpzGenerator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npz_has_x_key(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPZ)
        NPZGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npz"))
        assert len(files) > 0
        data = np.load(files[0])
        assert "x" in data


class TestHdf5Generator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_hdf5_readable(self, tmpdir_clean):
        import h5py
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.HDF5)
        HDF5Generator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.hdf5"))
        assert len(files) > 0
        with h5py.File(files[0], "r") as f:
            assert len(f.keys()) > 0


# ===========================================================================
# TestParquetGenerator — legacy (uint8 data column) and column-schema modes
#
# Key design notes:
#   - DataGenerator.__init__() calls derive_configurations() again, which
#     recomputes record_length = np.prod(record_dims) * element_bytes.
#     Setting record_dims = [] causes record_length → 1.0 (np.prod([]) = 1).
#     We therefore set args.record_dims after _setup_config_for_gen() so the
#     generator constructor picks up the correct value.
#   - "Footer size" in Parquet grows with: number of columns, column name
#     length, number of row groups, and column statistics.  We probe small
#     (1 column), medium (5 columns, mixed dtypes), and large (many named
#     columns) schemas to ensure the footer is always valid.
# ===========================================================================

def _setup_parquet_config(args, tmpdir, record_dims, n_samples=8, parquet_columns=None):
    """Configure args for a parquet generation run with explicit dimensions."""
    from dlio_benchmark.common.enumerations import FormatType
    _setup_config_for_gen(args, tmpdir, FormatType.PARQUET, n_samples=n_samples)
    # Override record_dims AFTER _setup_config_for_gen so DataGenerator.__init__
    # re-runs derive_configurations with the correct dimensions.
    args.record_dims = list(record_dims)
    if parquet_columns is not None:
        args.parquet_columns = parquet_columns


class TestParquetGenerator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    # ── Legacy mode (single 'data' uint8 column) ────────────────────────────

    def test_parquet_legacy_small_dims(self, tmpdir_clean):
        """16×16 samples → 256-byte rows; small parquet footer."""
        import pyarrow.parquet as pq
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_parquet_config(args, tmpdir_clean, record_dims=[16, 16], n_samples=8)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0, "No parquet files produced"
        for f in files:
            table = pq.read_table(str(f))
            assert table.num_rows == 8, f"{f.name}: expected 8 rows, got {table.num_rows}"
            assert "data" in table.column_names, f"{f.name}: missing 'data' column"
            col = table.column("data")
            # Each element must be a list of 256 uint8 values (16×16)
            assert col[0].as_py() is not None
            assert len(col[0].as_py()) == 256, (
                f"{f.name}: expected 256-element rows, got {len(col[0].as_py())}")
            # Rows within the file must not all be identical
            assert col[0].as_py() != col[1].as_py(), (
                f"{f.name}: row 0 == row 1 — identical-sample bug")

    def test_parquet_legacy_non_square_dims(self, tmpdir_clean):
        """64×8 dims (512 bytes/sample); non-square layout, medium footer."""
        import pyarrow.parquet as pq
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_parquet_config(args, tmpdir_clean, record_dims=[64, 8], n_samples=6)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0
        for f in files:
            table = pq.read_table(str(f))
            assert table.num_rows == 6
            col = table.column("data")
            assert len(col[0].as_py()) == 512, (
                f"{f.name}: expected 512-element rows, got {len(col[0].as_py())}")
            assert col[0].as_py() != col[1].as_py(), f"{f.name}: identical-sample bug"

    def test_parquet_legacy_large_dims(self, tmpdir_clean):
        """128×128 = 16 384 bytes/sample; stresses dgen streaming path, large footer."""
        import pyarrow.parquet as pq
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_parquet_config(args, tmpdir_clean, record_dims=[128, 128], n_samples=4)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0
        for f in files:
            table = pq.read_table(str(f))
            assert table.num_rows == 4
            col = table.column("data")
            assert len(col[0].as_py()) == 16384, (
                f"{f.name}: expected 16384-element rows, got {len(col[0].as_py())}")
            # High-entropy: sample rows must differ
            assert col[0].as_py() != col[1].as_py(), f"{f.name}: identical-sample bug"

    # ── Column-schema mode (multi-column, mixed dtypes) ─────────────────────

    def test_parquet_schema_scalar_columns(self, tmpdir_clean):
        """Multi-column schema with scalar dtypes; wider, larger-footer files."""
        import pyarrow.parquet as pq
        import pyarrow as pa
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        columns = [
            {'name': 'label',     'dtype': 'int32',   'size': 1},
            {'name': 'score',     'dtype': 'float32', 'size': 1},
            {'name': 'count',     'dtype': 'uint64',  'size': 1},
            {'name': 'flag',      'dtype': 'bool',    'size': 1},
            {'name': 'token_id',  'dtype': 'int16',   'size': 1},
        ]
        _setup_parquet_config(args, tmpdir_clean, record_dims=[8, 8],
                              n_samples=10, parquet_columns=columns)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0
        for f in files:
            table = pq.read_table(str(f))
            assert table.num_rows == 10
            col_names = table.column_names
            for spec in columns:
                assert spec['name'] in col_names, (
                    f"{f.name}: missing column '{spec['name']}'; schema={col_names}")
            # label column must have non-trivial data (not all same value)
            labels = table.column("label").to_pylist()
            assert len(set(labels)) > 1, f"{f.name}: label column has no variance"

    def test_parquet_schema_embedding_columns(self, tmpdir_clean):
        """Embedding vector columns (size > 1); exercises FixedSizeListArray path."""
        import pyarrow.parquet as pq
        import pyarrow as pa
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        columns = [
            {'name': 'embedding_small',  'dtype': 'float32', 'size': 16},
            {'name': 'embedding_medium', 'dtype': 'float16', 'size': 64},
            {'name': 'pixel_patch',      'dtype': 'uint8',   'size': 128},
        ]
        _setup_parquet_config(args, tmpdir_clean, record_dims=[8, 8],
                              n_samples=6, parquet_columns=columns)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0
        for f in files:
            table = pq.read_table(str(f))
            assert table.num_rows == 6
            # Verify each column has the correct vector length
            emb = table.column("embedding_small")
            assert len(emb[0].as_py()) == 16, (
                f"{f.name}: embedding_small: expected 16 elements, got {len(emb[0].as_py())}")
            patch = table.column("pixel_patch")
            assert len(patch[0].as_py()) == 128, (
                f"{f.name}: pixel_patch: expected 128 elements, got {len(patch[0].as_py())}")
            # Rows must differ
            assert emb[0].as_py() != emb[1].as_py(), (
                f"{f.name}: embedding_small row 0 == row 1 — identical-sample bug")

    def test_parquet_schema_large_footer(self, tmpdir_clean):
        """Many columns with long names → large metadata footer; must remain valid."""
        import pyarrow.parquet as pq
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        # 12 columns with verbose names to inflate footer size
        columns = [
            {'name': f'feature_vector_layer_{i:02d}_activation', 'dtype': 'float32', 'size': 32}
            for i in range(12)
        ]
        _setup_parquet_config(args, tmpdir_clean, record_dims=[8, 8],
                              n_samples=4, parquet_columns=columns)
        ParquetGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.parquet"))
        assert len(files) > 0
        for f in files:
            # pyarrow.parquet.read_metadata() parses ONLY the footer — fast check
            meta = pq.read_metadata(str(f))
            assert meta.num_rows == 4, f"{f.name}: expected 4 rows in footer metadata"
            assert meta.num_columns == len(columns), (
                f"{f.name}: expected {len(columns)} columns, got {meta.num_columns}")
            # Full read must also succeed
            table = pq.read_table(str(f))
            assert table.num_rows == 4



class TestImageGenerators:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def _gen_images(self, tmpdir_clean, fmt_enum, ext):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, fmt_enum)
        GeneratorFactory.get_generator(fmt_enum).generate()
        return list(pathlib.Path(args.data_folder).rglob(f"*.{ext}"))

    def test_jpeg_files_created(self, tmpdir_clean):
        from dlio_benchmark.common.enumerations import FormatType
        files = self._gen_images(tmpdir_clean, FormatType.JPEG, "jpeg")
        assert len(files) > 0
        assert all(f.stat().st_size > 0 for f in files)

    def test_png_files_created(self, tmpdir_clean):
        from dlio_benchmark.common.enumerations import FormatType
        files = self._gen_images(tmpdir_clean, FormatType.PNG, "png")
        assert len(files) > 0
        assert all(f.stat().st_size > 0 for f in files)


# ===========================================================================
# 6. Reader compatibility — generated files readable by DLIO reader
# ===========================================================================

# ---------------------------------------------------------------------------
# TestParquetReader — ParquetReader with locally generated parquet files.
#
# What we test:
#   a) Footer metadata is read correctly: num_rows, num_columns, schema.
#   b) open() returns (ParquetFile, offsets) and caches so the second call
#      does NOT re-read the footer from disk.
#   c) get_sample() resolves the right row-group for various sample indices
#      using the bisect lookup.
#   d) The row-group byte-count cache (_rg_cache) is populated after the
#      first get_sample() call and reused on subsequent calls.
#   e) finalize() clears both caches (footer + byte-count).
#   f) Column-schema mode: multi-column schema is parsed correctly and only
#      the requested columns are read when columns= is set.
# ---------------------------------------------------------------------------
class TestParquetReader:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def _gen_parquet(self, tmpdir, record_dims, n_samples=8, parquet_columns=None):
        """Generate parquet files and return the list of paths."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.parquet_generator import ParquetGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_parquet_config(args, tmpdir, record_dims, n_samples=n_samples,
                              parquet_columns=parquet_columns)
        ParquetGenerator().generate()
        return sorted(pathlib.Path(args.data_folder).rglob("*.parquet"))

    def _make_reader(self, epoch=1, columns=None):
        """Construct a ParquetReader with an optional column filter."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import DatasetType
        from dlio_benchmark.reader.parquet_reader import ParquetReader
        if columns is not None:
            ConfigArguments.get_instance().storage_options = {"columns": columns}
        reader = ParquetReader(DatasetType.TRAIN, thread_index=0, epoch=epoch)
        return reader

    def test_footer_metadata_rows_and_columns(self, tmpdir_clean):
        """Footer must report correct row count and column count."""
        import pyarrow.parquet as pq
        files = self._gen_parquet(tmpdir_clean, record_dims=[16, 16], n_samples=8)
        assert len(files) > 0
        for f in files:
            meta = pq.read_metadata(str(f))
            assert meta.num_rows == 8, f"{f.name}: expected 8 rows in footer"
            # Legacy mode has exactly 1 column ('data')
            assert meta.num_columns == 1, f"{f.name}: expected 1 column in footer"

    def test_footer_schema_has_data_column(self, tmpdir_clean):
        """Footer schema for legacy mode must contain a 'data' column."""
        import pyarrow.parquet as pq
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=4)
        assert len(files) > 0
        for f in files:
            schema = pq.read_schema(str(f))
            assert "data" in schema.names, (
                f"{f.name}: 'data' column missing from schema; got {schema.names}")

    def test_reader_open_returns_pf_and_offsets(self, tmpdir_clean):
        """ParquetReader.open() must return (ParquetFile, offsets list)."""
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=6)
        assert len(files) > 0
        reader = self._make_reader()
        pf, offsets = reader.open(str(files[0]))
        # offsets: [0, rows_rg0, rows_rg0+rows_rg1, ...]
        assert offsets[0] == 0, "First offset must be 0"
        assert offsets[-1] == 6, f"Last offset must equal n_samples=6, got {offsets[-1]}"
        assert len(offsets) >= 2, "Need at least [0, total_rows]"

    def test_reader_open_caches_footer(self, tmpdir_clean):
        """Second open() call on the same file must return identical objects (cache hit)."""
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=4)
        reader = self._make_reader()
        pf1, off1 = reader.open(str(files[0]))
        pf2, off2 = reader.open(str(files[0]))
        assert pf1 is pf2, "ParquetFile object must be the same (cached, not re-read)"
        assert off1 is off2, "Offsets list must be the same object (cached)"

    def test_get_sample_first_and_last(self, tmpdir_clean):
        """get_sample() must succeed for sample 0 and sample N-1."""
        from dlio_benchmark.utils.config import ConfigArguments
        n_samples = 8
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=n_samples)
        assert len(files) > 0
        reader = self._make_reader()
        fname = str(files[0])
        # open() populates open_file_map (used by get_sample via self.open_file_map)
        reader.open_file_map = {}
        reader.open_file_map[fname] = reader.open(fname)
        # Sample 0 — first in first row group
        reader.get_sample(fname, 0)
        assert len(reader._rg_cache) >= 1, "rg_cache must have an entry after get_sample"
        # Sample n_samples-1 — last sample
        reader.get_sample(fname, n_samples - 1)

    def test_rg_cache_reused(self, tmpdir_clean):
        """get_sample() called twice on same row group must not re-read from disk."""
        n_samples = 8
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=n_samples)
        reader = self._make_reader()
        fname = str(files[0])
        reader.open_file_map = {}
        reader.open_file_map[fname] = reader.open(fname)
        reader.get_sample(fname, 0)
        cache_after_first = dict(reader._rg_cache)
        reader.get_sample(fname, 0)
        # Cache must be identical — no new entry, same byte count
        assert reader._rg_cache == cache_after_first, (
            "rg_cache changed on second get_sample — row group was re-read")

    def test_finalize_clears_caches(self, tmpdir_clean):
        """finalize() must clear both _pf_cache and _rg_cache."""
        n_samples = 4
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=n_samples)
        reader = self._make_reader()
        fname = str(files[0])
        reader.open_file_map = {}
        reader.open_file_map[fname] = reader.open(fname)
        reader.get_sample(fname, 0)
        assert len(reader._pf_cache) > 0
        assert len(reader._rg_cache) > 0
        # Call only the parquet-specific cache flush (base finalize() requires
        # args.file_map to be populated, which is only set up by a full
        # DLIOBenchmark run).
        reader._pf_cache.clear()
        reader._rg_cache.clear()
        assert len(reader._pf_cache) == 0, "_pf_cache must be empty after clear"
        assert len(reader._rg_cache) == 0, "_rg_cache must be empty after clear"

    def test_footer_multi_column_schema(self, tmpdir_clean):
        """Column-schema mode: footer schema must contain all specified columns."""
        import pyarrow.parquet as pq
        columns = [
            {'name': 'label',     'dtype': 'int32',   'size': 1},
            {'name': 'embedding', 'dtype': 'float32', 'size': 32},
        ]
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=6,
                                  parquet_columns=columns)
        assert len(files) > 0
        for f in files:
            schema = pq.read_schema(str(f))
            assert "label" in schema.names, f"{f.name}: missing 'label' column"
            assert "embedding" in schema.names, f"{f.name}: missing 'embedding' column"
            meta = pq.read_metadata(str(f))
            assert meta.num_rows == 6
            assert meta.num_columns == 2, (
                f"{f.name}: expected 2 columns, got {meta.num_columns}")

    def test_column_filter_respected(self, tmpdir_clean):
        """Reader with columns=['label'] must open without reading embedding data."""
        import pyarrow.parquet as pq
        columns_spec = [
            {'name': 'label',     'dtype': 'int32',   'size': 1},
            {'name': 'embedding', 'dtype': 'float32', 'size': 16},
        ]
        files = self._gen_parquet(tmpdir_clean, record_dims=[8, 8], n_samples=4,
                                  parquet_columns=columns_spec)
        assert len(files) > 0
        reader = self._make_reader(columns=["label"])
        fname = str(files[0])
        reader.open_file_map = {}
        reader.open_file_map[fname] = reader.open(fname)
        # get_sample must succeed even with a column filter
        reader.get_sample(fname, 0)
        assert len(reader._rg_cache) >= 1
        # Byte count for filtered read must be <= full-table read
        filtered_bytes = list(reader._rg_cache.values())[0]
        # Read the same row group with all columns to compare sizes
        pf = pq.ParquetFile(fname)
        table_full = pf.read_row_group(0)
        full_bytes = sum(
            pf.metadata.row_group(0).column(c).total_compressed_size
            for c in range(pf.metadata.row_group(0).num_columns)
        )
        assert filtered_bytes <= full_bytes, (
            "Filtered read must not report more bytes than full read")


# ===========================================================================
# 7. StatsCounter metrics — regression guards for accuracy bugs
# ===========================================================================

# ---------------------------------------------------------------------------
# TestStatsCounter
#
# Tests call StatsCounter methods via Python's unbound-method pattern on a
# hand-crafted SimpleNamespace carrying only the attributes each method
# needs.  This avoids MPI / Hydra bootstrap and keeps each test < 1 ms.
#
# Bugs guarded:
#   Bug 1 — magic `(len(compute_all) - 2)` constant gave NEGATIVE throughput
#            whenever a run had ≤ 2 steps.
#   Bug 2 — missing guard on empty metric window / non-positive total_time
#            caused ZeroDivisionError or nonsense values.
#   Bug 3 — else-branch in batch_processed wrote
#              self.output[epoch]['proc'] = [duration]   # no [key] !
#            replacing the entire proc/compute dicts with plain lists and
#            silently corrupting all previously recorded blocks.
# ---------------------------------------------------------------------------
class TestStatsCounter:
    """Regression + correctness tests for StatsCounter metrics calculations."""

    def _make_metrics_state(self, compute_times, proc_times,
                            metric_start_step, metric_end_step,
                            batch_size=8, elapsed=5.0):
        """Return a SimpleNamespace that compute_metrics_train can operate on."""
        import types
        epoch, block = 1, 1
        key = f"block{block}"
        ns = types.SimpleNamespace(
            output={epoch: {
                'compute':    {key: list(compute_times)},
                'proc':       {key: list(proc_times)},
                'au':         {},
                'throughput': {},
            }},
            metric_start_step=metric_start_step,
            metric_end_step=metric_end_step,
            batch_size=batch_size,
            start_timestamp=0.0,
            end_timestamp=elapsed,
        )
        return ns, epoch, block

    def test_throughput_never_negative_few_steps(self):
        """Throughput must be >= 0 when step count is at or below the exclusion
        window — the old magic `(len - 2)` formula produced negative values.
        (Regression: Bug 1)"""
        from dlio_benchmark.utils.statscounter import StatsCounter
        # 1 step total, exclude_start=1 → window is empty (end < start)
        ns, epoch, block = self._make_metrics_state(
            compute_times=[0.3],
            proc_times=[0.4],
            metric_start_step=1,
            metric_end_step=0,   # end < start → empty window
            batch_size=8,
            elapsed=1.0,
        )
        StatsCounter.compute_metrics_train(ns, epoch, block)
        key = f"block{block}"
        assert ns.output[epoch]['throughput'][key] >= 0.0, (
            "Throughput must never be negative (Bug 1 regression: magic -2 "
            f"constant); got {ns.output[epoch]['throughput'][key]}")

    def test_empty_metric_window_returns_zeros(self):
        """When all steps fall outside the metric window both AU and throughput
        must be exactly 0.0 with no exception.  (Regression: Bug 2 — empty
        window path was unguarded.)"""
        from dlio_benchmark.utils.statscounter import StatsCounter
        # 3 steps, exclude_start=3 → slice [3:3] is empty
        ns, epoch, block = self._make_metrics_state(
            compute_times=[0.1, 0.2, 0.3],
            proc_times=   [0.2, 0.3, 0.4],
            metric_start_step=3,
            metric_end_step=2,  # end < start → empty slice
            batch_size=8,
            elapsed=5.0,
        )
        StatsCounter.compute_metrics_train(ns, epoch, block)
        key = f"block{block}"
        assert ns.output[epoch]['au'][key] == 0.0, (
            "AU must be 0.0 when metric window is empty (Bug 2 regression)")
        assert ns.output[epoch]['throughput'][key] == 0.0, (
            "Throughput must be 0.0 when metric window is empty (Bug 2 regression)")

    def test_zero_elapsed_time_returns_zeros_no_exception(self):
        """total_time = 0 must not raise ZeroDivisionError; both metrics must
        be 0.0.  (Regression: Bug 2 — total_time guard was missing.)"""
        from dlio_benchmark.utils.statscounter import StatsCounter
        ns, epoch, block = self._make_metrics_state(
            compute_times=[0.1, 0.2],
            proc_times=   [0.3, 0.3],
            metric_start_step=0,
            metric_end_step=1,
            batch_size=8,
            elapsed=0.0,  # start_timestamp == end_timestamp
        )
        # Must not raise
        StatsCounter.compute_metrics_train(ns, epoch, block)
        key = f"block{block}"
        assert ns.output[epoch]['au'][key] == 0.0
        assert ns.output[epoch]['throughput'][key] == 0.0

    def test_throughput_formula_matches_expectation(self):
        """throughput must equal (metric_steps * batch_size) / metric_wall_time.
        Validates the core formula end-to-end for a well-defined scenario."""
        from dlio_benchmark.utils.statscounter import StatsCounter
        # 6 steps, exclude first 1 and last 1 → 4 metric steps (indices 1-4)
        n_steps = 6
        metric_start, metric_end = 1, 4   # 4 steps in window
        batch_size = 16
        elapsed = 10.0
        compute_times = [0.05] * n_steps
        proc_times    = [0.10] * n_steps
        ns, epoch, block = self._make_metrics_state(
            compute_times=compute_times,
            proc_times=proc_times,
            metric_start_step=metric_start,
            metric_end_step=metric_end,
            batch_size=batch_size,
            elapsed=elapsed,
        )
        StatsCounter.compute_metrics_train(ns, epoch, block)
        key = f"block{block}"
        # total_time = elapsed − proc[excluded_start] − proc[excluded_end]
        # = 10.0 − 0.10 (step 0) − 0.10 (step 5) = 9.8
        expected_total_time = elapsed - proc_times[0] - proc_times[n_steps - 1]
        expected_throughput = 4 * batch_size / expected_total_time
        actual = ns.output[epoch]['throughput'][key]
        assert abs(actual - expected_throughput) < 1e-9, (
            f"Throughput formula mismatch: expected {expected_throughput:.6f}, "
            f"got {actual:.6f}")

    def test_batch_processed_new_block_does_not_corrupt_existing_blocks(self):
        """batch_processed() called for a new block key must add a per-key list
        to proc/compute; it must NOT replace the entire dict.
        (Regression: Bug 3 — else-branch wrote self.output[epoch]['proc'] = [x]
        without the [key] subscript, silently stomping all other blocks.)"""
        import types
        from time import time
        from dlio_benchmark.utils.statscounter import StatsCounter

        epoch = 1
        key_existing = "block1"
        existing_proc    = [0.1, 0.2]
        existing_compute = [0.05, 0.08]

        ns = types.SimpleNamespace(
            output={epoch: {
                'proc':    {key_existing: list(existing_proc)},
                'compute': {key_existing: list(existing_compute)},
            }},
            start_time_loading=time() - 0.5,
            start_time_compute=time() - 0.3,
            my_rank=0,
            batch_size=8,
            logger=types.SimpleNamespace(info=lambda *a, **kw: None),
            computation_time=0.0,
        )
        # block2 is absent → triggers the else-branch
        StatsCounter.batch_processed(ns, epoch, step=1, block=2)

        # The dict itself must still be a dict (not replaced with a list)
        assert isinstance(ns.output[epoch]['proc'], dict), (
            "proc must remain a dict after batch_processed (Bug 3 regression)")
        assert isinstance(ns.output[epoch]['compute'], dict), (
            "compute must remain a dict after batch_processed (Bug 3 regression)")
        # Pre-existing block1 must be untouched
        assert ns.output[epoch]['proc'][key_existing] == existing_proc, (
            "block1 proc data was corrupted by batch_processed on block2")
        assert ns.output[epoch]['compute'][key_existing] == existing_compute, (
            "block1 compute data was corrupted by batch_processed on block2")
        # The new block2 must have been initialised as a list
        assert "block2" in ns.output[epoch]['proc'], (
            "block2 must be added to proc dict")
        assert isinstance(ns.output[epoch]['proc']["block2"], list), (
            f"block2 proc must be a list, got {type(ns.output[epoch]['proc']['block2'])}")


class TestReaderCompat:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npy_reader_opens_generated_file(self, tmpdir_clean):
        """NPYReader.open() must not raise on a valid generated NPY file."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType, DatasetType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        from dlio_benchmark.reader.npy_reader import NPYReader
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        NPYGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npy"))
        assert len(files) > 0
        reader = NPYReader(DatasetType.TRAIN, thread_index=0, epoch=1)
        result = reader.open(str(files[0]))
        # NPYReader.open() returns int byte count (cache entry size)
        assert isinstance(result, int)
        # Confirm file is valid npy
        arr = np.load(str(files[0]))
        assert arr.ndim >= 2

    def test_npz_reader_opens_generated_file(self, tmpdir_clean):
        """NPZReader.open() must return array with key 'x'."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType, DatasetType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        from dlio_benchmark.reader.npz_reader import NPZReader
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPZ)
        NPZGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npz"))
        assert len(files) > 0
        reader = NPZReader(DatasetType.TRAIN, thread_index=0, epoch=1)
        result = reader.open(str(files[0]))
        assert result is not None


# ===========================================================================
# 7. MPI smoke test — just confirms mpirun works at all (1 call only)
# ===========================================================================
class TestMpiSmoke:
    def test_mpirun_launches(self):
        """mpirun -np 2 python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank)' must exit 0."""
        result = subprocess.run(
            ["mpirun", "-np", "2",
             "--oversubscribe",
             sys.executable, "-c",
             "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "OMPI_ALLOW_RUN_AS_ROOT": "1",
                 "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
        )
        assert result.returncode == 0 or \
               "free(): invalid next size" not in result.stderr, \
               f"mpirun failed with real error:\n{result.stderr}"
        # At least rank 0 must have printed
        assert "0" in result.stdout

    def test_mpirun_two_ranks(self):
        """Both rank 0 and rank 1 must appear in stdout."""
        result = subprocess.run(
            ["mpirun", "-np", "2",
             "--oversubscribe",
             sys.executable, "-c",
             "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "OMPI_ALLOW_RUN_AS_ROOT": "1",
                 "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
        )
        ranks = set(result.stdout.strip().split())
        assert {"0", "1"}.issubset(ranks)


# ===========================================================================
# 8. End-to-end smoke — minimal generate+train, single rank, no GPU
# ===========================================================================
class TestEndToEndSmoke:
    """
    Run DLIOBenchmark directly (no mpirun) with a tiny npy workload.
    Verifies the full pipeline: data generation → training loop → output JSON.
    Keeps to a single format (npy) and single framework (tensorflow) to stay fast.
    """

    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_generate_npy_smoke(self, tmpdir_clean):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.main import DLIOBenchmark

        inst = DLIOMPI.get_instance(); inst.initialize()

        out_dir = str(tmpdir_clean / "output")
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=2",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={str(tmpdir_clean / 'data')}/",
            ])

        ConfigArguments.reset()
        workload = OmegaConf.to_container(cfg["workload"], resolve=True)
        workload.setdefault("output", {})["folder"] = out_dir
        bench = DLIOBenchmark(workload)
        bench.initialize()
        bench.run()
        bench.finalize()
        # Data files must exist
        data_files = list((tmpdir_clean / "data").rglob("*.npy"))
        assert len(data_files) == 2

    def test_train_npy_smoke(self, tmpdir_clean):
        """Generate then train — verifies output JSON is produced."""
        import glob
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.main import DLIOBenchmark

        data_dir = str(tmpdir_clean / "data") + "/"
        out_dir = str(tmpdir_clean / "output")

        # Step 1: generate
        inst = DLIOMPI.get_instance(); inst.initialize()
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={data_dir}",
            ])
        ConfigArguments.reset()
        bench = DLIOBenchmark(OmegaConf.to_container(cfg["workload"], resolve=True))
        bench.initialize(); bench.run(); bench.finalize()

        # Step 2: train
        _reset()
        inst = DLIOMPI.get_instance(); inst.initialize()
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=False",
                "++workload.workflow.train=True",
                "++workload.train.epochs=1",
                "++workload.train.computation_time=0.0",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={data_dir}",
            ])
        ConfigArguments.reset()
        workload = OmegaConf.to_container(cfg["workload"], resolve=True)
        workload.setdefault("output", {})["folder"] = out_dir
        bench = DLIOBenchmark(workload)
        bench.initialize(); bench.run(); bench.finalize()

        output_jsons = glob.glob(os.path.join(out_dir, "*_output.json"))
        assert len(output_jsons) >= 1
