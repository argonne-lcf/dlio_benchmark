from dlio_benchmark.data_loader.synthetic_data_loader import SyntheticDataLoader


def test_getitem_returns_precreated_zero_batch():
    loader = SyntheticDataLoader.__new__(SyntheticDataLoader)
    zero_batch = object()
    loader.zero_batch = zero_batch

    assert loader.getitem() is zero_batch
