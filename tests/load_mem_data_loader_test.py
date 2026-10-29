"""Regression tests for iteration over preloaded batches."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dlio_benchmark.data_loader.load_mem_data_loader import LoadMemDataLoader


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = LoadMemDataLoader.__new__(LoadMemDataLoader)
    loader._args = SimpleNamespace(my_rank=0, iter_time=0)
    loader.logger = logging.getLogger("DLIO")
    loader._preloaded_batches = []
    loader._torch_loader = Mock()
    return loader


def test_next_warns_when_preloading_produces_no_batches(loader, caplog):
    loader._torch_loader.next.return_value = iter(())

    with caplog.at_level(logging.WARNING, logger="DLIO"):
        assert list(loader.next()) == []

    loader._torch_loader.read.assert_called_once_with()
    loader._torch_loader.next.assert_called_once_with()
    assert caplog.record_tuples == [
        ("DLIO", logging.WARNING, "No preloaded batches are available for iteration.")
    ]


@pytest.mark.parametrize("already_preloaded", [False, True])
def test_next_yields_batches_without_warning(loader, caplog, already_preloaded):
    batches = [torch.zeros((2, 3)), torch.ones((2, 3))]
    if already_preloaded:
        loader._preloaded_batches = batches.copy()
    else:
        loader._torch_loader.next.return_value = iter(batches)

    with caplog.at_level(logging.WARNING, logger="DLIO"):
        for _ in range(2):
            result = list(loader.next())
            assert len(result) == len(batches)
            assert all(actual is expected for actual, expected in zip(result, batches))

    assert caplog.record_tuples == []
    if already_preloaded:
        loader._torch_loader.read.assert_not_called()
        loader._torch_loader.next.assert_not_called()
    else:
        loader._torch_loader.read.assert_called_once_with()
        loader._torch_loader.next.assert_called_once_with()
