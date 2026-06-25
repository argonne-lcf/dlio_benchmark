"""
Regression tests for ObjStoreLibStorage._preflight (mlcommons/storage#472).

Background
==========

mlcommons/storage#472 (FileSystemGuy diagnosis) identified three silent
misconfiguration paths in the object-storage init in
`dlio_benchmark/storage/obj_store_lib.py`:

  1. Missing AWS_ACCESS_KEY_ID / SECRET surfaces only at the first
     put_data() inside an upload worker.  The data_generator submission
     loop kept queueing more uploads (each retaining its BytesIO payload)
     until OOM — see #472 / #392 / #393 / #504.
  2. s3torchconnector's S3Client is lazy.  When endpoint is None and
     AWS_ENDPOINT_URL is also unset, it silently routes to real AWS S3 —
     wrong bucket, real transfer cost — instead of erroring.
  3. A typo in storage_root (the bucket name) only surfaces as the first
     NoSuchBucket on PUT, after MPI + worker pool spin-up.

This module verifies that the new `_preflight()` method in
`ObjStoreLibStorage` turns each silent failure into a loud, explicit,
single error at construction time.

Test strategy: `_preflight()` is a self-contained method that reads a
small set of attributes (storage_library, access_key_id, secret_access_key,
endpoint, namespace.name, plus either self._s3dlio or self.s3_client).
We create an instance via __new__ (bypassing the framework-dependent
__init__), set just those attributes, and call _preflight() directly.
No mocking of the DLIO framework registry, no real network, no library
installs required.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from dlio_benchmark.storage.obj_store_lib import ObjStoreLibStorage


# ─── Test fixture: a partially-constructed storage instance ──────────────


def _make_partial_storage(
    storage_library,
    *,
    access_key_id="test-ak",
    secret_access_key="test-sk",
    endpoint="http://test-endpoint:9000",
    bucket="test-bucket",
    uri_scheme="s3",
    backend_stub=None,
):
    """Build an ObjStoreLibStorage with only the attributes _preflight()
    reads — bypassing the full __init__ (which would import DLIO framework
    state and the real backend).

    Returns the instance.  Caller invokes ``inst._preflight()`` to run the
    method under test.
    """
    inst = ObjStoreLibStorage.__new__(ObjStoreLibStorage)
    inst.storage_library = storage_library
    inst.access_key_id = access_key_id
    inst.secret_access_key = secret_access_key
    inst.endpoint = endpoint
    inst.uri_scheme = uri_scheme
    inst.namespace = SimpleNamespace(name=bucket)

    # Per-library backend stub.  Each library wires a different attribute,
    # so we stash the stub on whichever attribute _preflight reads from.
    if storage_library == "s3dlio":
        inst._s3dlio = backend_stub if backend_stub is not None else mock.MagicMock()
        # Happy-path default: empty bucket → empty list, no exception.
        if not isinstance(inst._s3dlio.list, mock.Mock):
            inst._s3dlio.list = mock.MagicMock(return_value=[])
    elif storage_library in ("s3torchconnector", "minio"):
        inst.s3_client = backend_stub if backend_stub is not None else mock.MagicMock()
        if storage_library == "s3torchconnector" and not isinstance(
            inst.s3_client.list_objects, mock.Mock
        ):
            inst.s3_client.list_objects = mock.MagicMock(return_value=iter([]))
        if storage_library == "minio" and not isinstance(
            inst.s3_client.bucket_exists, mock.Mock
        ):
            inst.s3_client.bucket_exists = mock.MagicMock(return_value=True)
    else:
        raise AssertionError(f"unsupported library: {storage_library}")

    return inst


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Each test starts with a clean view of the env vars _preflight reads."""
    for name in (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


# ─── 1. Credentials check (library-agnostic) ─────────────────────────────


class TestPreflightCredentialsCheck:
    def test_missing_access_key_raises_valueerror(self):
        inst = _make_partial_storage("s3dlio", access_key_id=None)
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            inst._preflight()

    def test_missing_secret_key_raises_valueerror(self):
        inst = _make_partial_storage("s3dlio", secret_access_key=None)
        with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
            inst._preflight()

    def test_empty_access_key_raises_valueerror(self):
        inst = _make_partial_storage("s3dlio", access_key_id="")
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            inst._preflight()

    def test_empty_secret_key_raises_valueerror(self):
        inst = _make_partial_storage("s3dlio", secret_access_key="")
        with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
            inst._preflight()

    def test_credential_error_explains_oom_link(self):
        """The headline reason for failing here: silent failure → OOM.
        The error message should make that consequence visible to the
        operator so they don't get blindsided when retrying."""
        inst = _make_partial_storage("s3dlio", access_key_id=None)
        with pytest.raises(ValueError) as exc_info:
            inst._preflight()
        msg = str(exc_info.value)
        assert "OOM" in msg or "data_generator" in msg, (
            "credential error must explain the OOM consequence; "
            f"actual message: {msg!r}"
        )

    def test_credentials_check_runs_for_minio_too(self):
        """The credential check is library-agnostic — every backend needs it."""
        inst = _make_partial_storage("minio", access_key_id=None)
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            inst._preflight()

    def test_credentials_check_runs_for_s3torchconnector_too(self):
        inst = _make_partial_storage("s3torchconnector", access_key_id=None)
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            inst._preflight()


# ─── 2. s3torchconnector silent-route-to-AWS guard ───────────────────────


class TestPreflightS3TorchconnectorSilentRouteGuard:
    """The headline #472 bug: s3torchconnector silently using real AWS S3."""

    def test_endpoint_none_and_env_unset_raises(self):
        inst = _make_partial_storage("s3torchconnector", endpoint=None)
        with pytest.raises(ValueError, match="endpoint"):
            inst._preflight()

    def test_endpoint_empty_and_env_unset_raises(self):
        inst = _make_partial_storage("s3torchconnector", endpoint="")
        with pytest.raises(ValueError, match="endpoint"):
            inst._preflight()

    def test_endpoint_none_but_env_set_allows_bucket_check(self, monkeypatch):
        """If AWS_ENDPOINT_URL is set in the env, the silent-route guard
        should not fire — the user has expressed an explicit (env-driven)
        endpoint choice."""
        monkeypatch.setenv(
            "AWS_ENDPOINT_URL", "http://fallback-via-env:9000"
        )
        inst = _make_partial_storage("s3torchconnector", endpoint=None)
        # Should not raise the silent-route guard.  The downstream bucket
        # check is stubbed to succeed by the fixture's default.
        inst._preflight()

    def test_silent_route_error_message_explains_aws_risk(self):
        inst = _make_partial_storage("s3torchconnector", endpoint=None)
        with pytest.raises(ValueError) as exc_info:
            inst._preflight()
        msg = str(exc_info.value)
        assert "AWS" in msg and "silent" in msg.lower(), (
            "guard error must explain the silent-route-to-real-AWS risk; "
            f"actual message: {msg!r}"
        )

    def test_silent_route_guard_does_not_fire_for_s3dlio(self):
        """s3dlio reads AWS_ENDPOINT_URL itself; we do not need to gate it.
        Endpoint can be None for s3dlio and the bucket reachability call
        will surface any actual problem."""
        inst = _make_partial_storage("s3dlio", endpoint=None)
        # Should not raise — bucket check is stubbed to succeed.
        inst._preflight()

    def test_silent_route_guard_does_not_fire_for_minio(self):
        """minio fails loudly on endpoint=None — no silent guard needed."""
        inst = _make_partial_storage("minio", endpoint=None)
        # Should not raise on the s3torchconnector guard — bucket check
        # stub succeeds.
        inst._preflight()


# ─── 3. Bucket reachability (library-specific) ───────────────────────────


class TestPreflightBucketReachability:
    """Each library's preflight calls a different lightweight 'ping' API."""

    def test_s3dlio_dispatch_failure_raises_connectionerror(self):
        backend = mock.MagicMock()
        backend.list.side_effect = Exception(
            "dispatch failure (connection timeout) — check AWS_ENDPOINT_URL"
        )
        inst = _make_partial_storage("s3dlio", backend_stub=backend)
        with pytest.raises(ConnectionError, match="cannot reach bucket"):
            inst._preflight()

    def test_s3dlio_empty_bucket_passes(self):
        # Returning [] is "bucket exists, just empty" — preflight passes.
        inst = _make_partial_storage("s3dlio")
        inst._preflight()  # must not raise

    def test_s3dlio_preflight_uses_only_selected_library(self):
        """Preflight must dispatch on storage_library and call only the
        selected backend, not any other library's API."""
        s3dlio_stub = mock.MagicMock()
        s3dlio_stub.list.return_value = []
        inst = _make_partial_storage("s3dlio", backend_stub=s3dlio_stub)
        inst._preflight()
        s3dlio_stub.list.assert_called_once()
        # The call should target the bucket root via the configured scheme.
        call_arg = s3dlio_stub.list.call_args[0][0]
        assert call_arg.startswith("s3://"), call_arg
        assert "test-bucket" in call_arg, call_arg

    def test_minio_bucket_does_not_exist_raises_connectionerror(self):
        backend = mock.MagicMock()
        backend.bucket_exists.return_value = False
        inst = _make_partial_storage("minio", backend_stub=backend)
        with pytest.raises(ConnectionError, match="cannot reach bucket"):
            inst._preflight()

    def test_minio_connection_error_raises_connectionerror(self):
        backend = mock.MagicMock()
        backend.bucket_exists.side_effect = Exception(
            "connection refused"
        )
        inst = _make_partial_storage("minio", backend_stub=backend)
        with pytest.raises(ConnectionError, match="cannot reach bucket"):
            inst._preflight()

    def test_minio_preflight_uses_only_minio_api(self):
        backend = mock.MagicMock()
        backend.bucket_exists.return_value = True
        inst = _make_partial_storage("minio", backend_stub=backend)
        inst._preflight()
        backend.bucket_exists.assert_called_once_with("test-bucket")

    def test_s3torchconnector_auth_failure_raises_connectionerror(self):
        backend = mock.MagicMock()
        backend.list_objects.side_effect = Exception("SignatureDoesNotMatch")
        inst = _make_partial_storage("s3torchconnector", backend_stub=backend)
        with pytest.raises(ConnectionError, match="cannot reach bucket"):
            inst._preflight()

    def test_s3torchconnector_empty_bucket_passes(self):
        backend = mock.MagicMock()
        backend.list_objects.return_value = iter([])  # empty bucket
        inst = _make_partial_storage("s3torchconnector", backend_stub=backend)
        inst._preflight()  # must not raise

    def test_s3torchconnector_preflight_uses_only_list_objects(self):
        backend = mock.MagicMock()
        backend.list_objects.return_value = iter([])
        inst = _make_partial_storage("s3torchconnector", backend_stub=backend)
        inst._preflight()
        backend.list_objects.assert_called_once_with("test-bucket")

    def test_connectionerror_message_includes_diagnostic_hints(self):
        backend = mock.MagicMock()
        backend.bucket_exists.side_effect = Exception("connection refused")
        inst = _make_partial_storage("minio", backend_stub=backend)
        with pytest.raises(ConnectionError) as exc_info:
            inst._preflight()
        msg = str(exc_info.value)
        for needle in ("AWS_ENDPOINT_URL", "bucket", "credentials"):
            assert needle in msg, (
                f"diagnostic message must mention {needle!r}; "
                f"actual message: {msg!r}"
            )


# ─── 4. Happy path: all three libraries succeed end-to-end ───────────────


class TestPreflightHappyPath:
    def test_s3dlio_happy_path(self):
        inst = _make_partial_storage("s3dlio")
        inst._preflight()  # must not raise

    def test_s3torchconnector_happy_path(self):
        inst = _make_partial_storage("s3torchconnector")
        inst._preflight()  # must not raise

    def test_minio_happy_path(self):
        inst = _make_partial_storage("minio")
        inst._preflight()  # must not raise
