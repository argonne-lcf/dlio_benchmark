"""Regression tests for dlio_sampler — per-rank equality and Sampler contract.

Pins the fix for mlcommons/storage#455: inter-epoch deadlock caused by
math.ceil(N/size) producing unequal per-rank batch counts when N is not a
multiple of comm_size.
"""

from dlio_benchmark.data_loader.torch_data_loader import dlio_sampler


def test_dlio_sampler_equalizes_uneven_rank_counts():
    """Every rank gets the same per-rank shard, with trailing samples dropped.

    The original ceil+clamp produced [15,15,15,15,15,15,10] for (N=100, size=7);
    the deadlock comes from the last rank doing fewer batches under drop_last.
    """
    total = 100
    size = 7
    batch_size = 3

    per_rank_samples = [
        len(list(dlio_sampler(rank, size, total, epochs=1)))
        for rank in range(size)
    ]
    per_rank_batches = [n // batch_size for n in per_rank_samples]

    assert per_rank_samples == [14] * 7
    assert per_rank_batches == [4] * 7
    assert total - sum(per_rank_samples) == 2


def test_dlio_sampler_len_matches_iterator_length():
    """PyTorch Sampler contract: len(sampler) == len(list(iter(sampler)))."""
    total = 100
    size = 7

    for rank in range(size):
        sampler = dlio_sampler(rank, size, total, epochs=1)
        assert len(sampler) == len(list(iter(sampler)))


def test_dlio_sampler_even_division_unchanged():
    """When N is a multiple of size, behavior is identical to the old impl."""
    total = 100
    size = 10

    per_rank_samples = [
        len(list(dlio_sampler(rank, size, total, epochs=1)))
        for rank in range(size)
    ]
    assert per_rank_samples == [10] * 10
    assert total - sum(per_rank_samples) == 0
