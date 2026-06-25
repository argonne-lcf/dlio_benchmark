"""
Regression test for mlcommons/storage#504.

Background
==========

Issue #504 reports that ``--params dataset.skip_listing=True`` (and the
lowercase ``true`` variant) had no effect at runtime: rank 0 still spent
20+ minutes listing 50M objects on retinanet.  Investigation found that
``ConfigArguments`` declares ``skip_listing: bool = False`` and
``listing_validation_interval: int = 1000``, but ``LoadConfig()`` never
copied these fields out of ``config['dataset']`` into ``args``.  The Hydra
override reached the config tree, but the args were never updated, so the
skip_listing fast path in ``DLIOBenchmark.__init__`` never ran.

This test guards against that exact regression: both fields must propagate
through ``LoadConfig()`` for the common Hydra value shapes, including the
string forms that ``++workload.dataset.skip_listing=...`` produces on the
CLI.
"""

import pytest
from omegaconf import OmegaConf

from dlio_benchmark.utils.config import ConfigArguments, LoadConfig
from dlio_benchmark.utils.utility import DLIOMPI


@pytest.fixture(scope='module', autouse=True)
def _mpi_init():
    """ConfigArguments.__init__ calls DLIOMPI.size() — initialize it once."""
    DLIOMPI.get_instance().initialize()


def _fresh_args():
    """ConfigArguments is a singleton; reset the fields we touch each test."""
    args = ConfigArguments.get_instance()
    args.skip_listing = False
    args.listing_validation_interval = 1000
    return args


def test_skip_listing_propagates_when_yaml_sets_python_bool_true():
    """A YAML-native ``true`` reaches ``args.skip_listing`` as True."""
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'skip_listing': True}})
    LoadConfig(args, cfg)
    assert args.skip_listing is True


def test_skip_listing_propagates_when_yaml_sets_python_bool_false():
    """An explicit ``false`` must still set the field (not just rely on default)."""
    args = _fresh_args()
    args.skip_listing = True  # start opposite of expected end state
    cfg = OmegaConf.create({'dataset': {'skip_listing': False}})
    LoadConfig(args, cfg)
    assert args.skip_listing is False


def test_skip_listing_string_true_lowercase_coerces_to_true():
    """The exact form Hydra produces from ``++workload.dataset.skip_listing=true``.

    austingnanaraj's command in #504 used this lowercase form and reported no
    effect — this assertion is the direct regression guard.
    """
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'skip_listing': 'true'}})
    LoadConfig(args, cfg)
    assert args.skip_listing is True


def test_skip_listing_string_true_titlecase_coerces_to_true():
    """``True`` (Python repr form) must also coerce."""
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'skip_listing': 'True'}})
    LoadConfig(args, cfg)
    assert args.skip_listing is True


def test_skip_listing_string_false_coerces_to_false():
    args = _fresh_args()
    args.skip_listing = True
    cfg = OmegaConf.create({'dataset': {'skip_listing': 'false'}})
    LoadConfig(args, cfg)
    assert args.skip_listing is False


def test_skip_listing_string_garbage_coerces_to_false():
    """Anything other than truthy strings should not enable skip_listing."""
    args = _fresh_args()
    args.skip_listing = True
    cfg = OmegaConf.create({'dataset': {'skip_listing': 'maybe'}})
    LoadConfig(args, cfg)
    assert args.skip_listing is False


def test_skip_listing_string_int_one_coerces_to_true():
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'skip_listing': '1'}})
    LoadConfig(args, cfg)
    assert args.skip_listing is True


def test_skip_listing_absent_field_leaves_default():
    """If the YAML/Hydra config does not set skip_listing, args keeps its default."""
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'num_files_train': 100}})
    LoadConfig(args, cfg)
    assert args.skip_listing is False  # ConfigArguments default


def test_listing_validation_interval_propagates_int():
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'listing_validation_interval': 50000}})
    LoadConfig(args, cfg)
    assert args.listing_validation_interval == 50000


def test_listing_validation_interval_string_coerces_to_int():
    """Hydra CLI overrides arrive as strings; integer fields must coerce."""
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {'listing_validation_interval': '50000'}})
    LoadConfig(args, cfg)
    assert args.listing_validation_interval == 50000


def test_listing_validation_interval_absent_field_leaves_default():
    args = _fresh_args()
    cfg = OmegaConf.create({'dataset': {}})
    LoadConfig(args, cfg)
    assert args.listing_validation_interval == 1000  # ConfigArguments default
