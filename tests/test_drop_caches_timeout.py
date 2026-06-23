"""Tests for the DLIO_DROP_CACHES_TIMEOUT env var parsing (mlcommons/storage #487).

The helper under test is small and pure: it reads an env mapping and returns an
integer >= 1. We accept a wide range of bad input gracefully (collapse to the
default) because the underlying `subprocess.run(timeout=...)` API rejects 0 /
negative values at call time, and a typo in an operator's environment must not
crash DLIO.
"""

import pytest

from dlio_benchmark.main import (
    _DROP_CACHES_TIMEOUT_DEFAULT_SECONDS,
    _resolve_drop_caches_timeout,
)


_DEFAULT = _DROP_CACHES_TIMEOUT_DEFAULT_SECONDS


class TestResolveDropCachesTimeout:
    """Unit tests for _resolve_drop_caches_timeout()."""

    def test_default_when_env_var_absent(self):
        """No env var set → default."""
        assert _resolve_drop_caches_timeout(env={}) == _DEFAULT

    def test_default_when_env_var_empty(self):
        """Empty string → default."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": ""}) == _DEFAULT

    def test_default_when_env_var_whitespace_only(self):
        """All-whitespace value → default."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": "   "}) == _DEFAULT

    @pytest.mark.parametrize("raw", ["not-a-number", "30s", "30.5", "30 ", "5,000", "nan", "True"])
    def test_default_when_value_is_unparseable(self, raw):
        """Non-integer text → default (no crash)."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": raw}) == _DEFAULT

    @pytest.mark.parametrize("raw, expected", [
        ("1", 1),
        ("60", 60),
        ("300", 300),
        ("7200", 7200),
    ])
    def test_valid_integer_override(self, raw, expected):
        """Valid positive integer → that value."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": raw}) == expected

    def test_whitespace_around_valid_integer_is_stripped(self):
        """Leading/trailing whitespace around an int is OK."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": "  120\t"}) == 120

    @pytest.mark.parametrize("raw", ["0", "-1", "-300"])
    def test_zero_and_negative_clamped_to_one(self, raw):
        """0 and negative values are clamped to 1 (subprocess.run rejects 0/<0)."""
        assert _resolve_drop_caches_timeout(env={"DLIO_DROP_CACHES_TIMEOUT": raw}) == 1

    def test_unrelated_env_vars_ignored(self):
        """Other DLIO env vars don't affect the result."""
        env = {
            "DLIO_OUTPUT_FOLDER": "/tmp",
            "DLIO_MAX_AUTO_THREADS": "4",
            "DLIO_DROP_CACHES_TIMEOUT_TYPO": "300",
        }
        assert _resolve_drop_caches_timeout(env=env) == _DEFAULT

    def test_default_constant_matches_storage_391_history(self):
        """Sanity check: the default must remain 30s.

        Lowering it would re-introduce the slow-flush regression
        (mlcommons/storage #487).  Raising it without coordination would shift
        the original sudo-prompt hang window (#391).  This test is a
        speed-bump that forces a deliberate choice if either changes.
        """
        assert _DEFAULT == 30

    def test_defaults_to_os_environ_when_env_arg_omitted(self, monkeypatch):
        """No env= arg → reads from os.environ."""
        monkeypatch.setenv("DLIO_DROP_CACHES_TIMEOUT", "180")
        assert _resolve_drop_caches_timeout() == 180

    def test_defaults_to_os_environ_when_var_unset(self, monkeypatch):
        """No env= arg, var unset → default."""
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        assert _resolve_drop_caches_timeout() == _DEFAULT
