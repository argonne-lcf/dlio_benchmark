"""
Tests for _preflight() behavior with direct:// and file:// URI schemes.

For local-filesystem schemes, _preflight() must NOT require S3 credentials.
Instead it verifies the storage-root directory exists and is readable+writable,
providing an early, clear error for misconfigured paths before MPI workers start.
"""

import os
import stat
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_storage(uri_scheme, bucket, storage_library="s3dlio",
                  access_key="key", secret_key="secret"):
    """Return a partially-constructed ObjStoreLibStorage for preflight tests only."""
    from dlio_benchmark.storage.obj_store_lib import ObjStoreLibStorage

    obj = object.__new__(ObjStoreLibStorage)
    # Minimal attributes _preflight() inspects
    obj.uri_scheme = uri_scheme
    obj.namespace = MagicMock()
    obj.namespace.name = bucket
    obj.storage_library = storage_library
    obj.access_key_id = access_key
    obj.secret_access_key = secret_key
    obj.endpoint = "https://localhost:9000"
    obj._args = MagicMock()
    return obj


# ---------------------------------------------------------------------------
# direct:// — happy path (dir exists and is readable+writable)
# ---------------------------------------------------------------------------

class TestDirectSchemeHappyPath:
    def test_direct_passes_when_dir_exists_and_writable(self, tmp_path):
        obj = _make_storage('direct', str(tmp_path))
        obj._preflight()  # must not raise

    def test_file_passes_when_dir_exists_and_writable(self, tmp_path):
        obj = _make_storage('file', str(tmp_path))
        obj._preflight()  # must not raise

    def test_does_not_check_s3_credentials_for_direct(self, tmp_path):
        """No S3 credentials needed — preflight must not raise for empty keys."""
        obj = _make_storage('direct', str(tmp_path), access_key='', secret_key='')
        obj._preflight()  # must not raise


# ---------------------------------------------------------------------------
# direct:// — directory does not exist
# ---------------------------------------------------------------------------

class TestDirectSchemeMissingDir:
    def test_raises_valueerror_when_dir_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent_subdir")
        obj = _make_storage('direct', missing)
        with pytest.raises(ValueError, match="does not exist or is not a directory"):
            obj._preflight()

    def test_raises_valueerror_for_file_scheme_missing_dir(self, tmp_path):
        missing = str(tmp_path / "ghost")
        obj = _make_storage('file', missing)
        with pytest.raises(ValueError, match="does not exist or is not a directory"):
            obj._preflight()


# ---------------------------------------------------------------------------
# direct:// — directory not writable
# ---------------------------------------------------------------------------

class TestDirectSchemeNotWritable:
    def test_raises_valueerror_when_dir_not_writable(self, tmp_path):
        # Make dir read-only
        tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            obj = _make_storage('direct', str(tmp_path))
            with pytest.raises(ValueError, match="not readable/writable"):
                obj._preflight()
        finally:
            tmp_path.chmod(stat.S_IRWXU)  # restore so pytest can clean up


# ---------------------------------------------------------------------------
# s3:// still requires credentials
# ---------------------------------------------------------------------------

class TestS3SchemeCredentialGuard:
    def test_s3_raises_valueerror_on_missing_credentials(self, tmp_path):
        """The credential guard must still fire for s3:// scheme."""
        obj = _make_storage('s3', 'my-bucket', access_key='', secret_key='')
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            obj._preflight()
