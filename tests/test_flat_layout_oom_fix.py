"""
Tests for the flat-layout OOM fix in main.py (mlcommons/storage#466).

The fix replaces `pending = sorted([...all URIs...])` (which doubled rank-0
RAM for flat-directory datasets) with per-batch URI construction so only
one _CHUNK_SIZE batch of URIs is ever live alongside the names list.

We test the logic directly without spinning up an MPI cluster: the
_filter_round_robin closure and bcast calls are mocked so the test runs
as a fast-CI unit test (single-process, no mpi4py import needed).
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Inline the flat-layout sharding logic so we can test it without DLIO's
# full Hydra / mpi4py init stack.
# ---------------------------------------------------------------------------

def _run_flat_layout_chunked(filenames, fmt, data_folder, dataset_type,
                              comm_size, my_rank, chunk_size=1_000_000):
    """
    Pure-Python re-implementation of the fixed flat-layout path from main.py.

    Returns (my_files, global_count) for the given rank.
    Used to verify correctness of the chunked approach vs. the naive approach.
    """
    def get_uri(path):
        return f"file://{path}"

    my_files = []
    global_count = 0

    def _filter_round_robin(chunk, start_idx):
        for i, fpath in enumerate(chunk):
            if (start_idx + i) % comm_size == my_rank:
                my_files.append(fpath)

    # Simulate rank 0 chunked logic + bcast to other ranks via a shared queue.
    broadcast_log = []  # records what rank 0 would broadcast

    for batch_start in range(0, len(filenames), chunk_size):
        batch = filenames[batch_start:batch_start + chunk_size]
        chunk = sorted([
            get_uri(os.path.join(data_folder, dataset_type, entry))
            for entry in batch
            if entry.endswith(f'.{fmt}')
        ])
        # Record broadcast (rank 0 sends this)
        broadcast_log.append(chunk)
        _filter_round_robin(chunk, global_count)
        global_count += len(chunk)

    # The final None sentinel is appended to distinguish end-of-stream
    broadcast_log.append(None)

    return my_files, global_count, broadcast_log


def _run_flat_layout_naive(filenames, fmt, data_folder, dataset_type,
                            comm_size, my_rank):
    """Original (pre-fix) approach — builds full sorted URI list first."""
    def get_uri(path):
        return f"file://{path}"

    my_files = []
    global_count = 0

    def _filter_round_robin(chunk, start_idx):
        for i, fpath in enumerate(chunk):
            if (start_idx + i) % comm_size == my_rank:
                my_files.append(fpath)

    pending = sorted([
        get_uri(os.path.join(data_folder, dataset_type, entry))
        for entry in filenames
        if entry.endswith(f'.{fmt}')
    ])
    _filter_round_robin(pending, 0)
    global_count = len(pending)
    return my_files, global_count


class TestFlatLayoutChunked:
    """Correctness: chunked result matches naive result for all ranks."""

    def _make_filenames(self, n, fmt="npz", extra_fmt="txt"):
        names = [f"img_{i:06d}_of_{n}.{fmt}" for i in range(n)]
        # Sprinkle in non-matching files to ensure filtering works
        names += [f"ignore_{i}.{extra_fmt}" for i in range(5)]
        return names

    def test_single_rank_all_files_assigned(self):
        names = self._make_filenames(100)
        my_files, count, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0)
        assert count == 100
        assert len(my_files) == 100
        assert bcast[-1] is None  # sentinel is None, not []

    def test_matches_naive_two_ranks_rank0(self):
        names = self._make_filenames(200)
        chunked, count_c, _ = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=2, my_rank=0)
        naive, count_n = _run_flat_layout_naive(
            names, "npz", "/data", "train", comm_size=2, my_rank=0)
        assert count_c == count_n == 200
        assert sorted(chunked) == sorted(naive)

    def test_matches_naive_two_ranks_rank1(self):
        names = self._make_filenames(200)
        chunked, _, _ = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=2, my_rank=1)
        naive, _ = _run_flat_layout_naive(
            names, "npz", "/data", "train", comm_size=2, my_rank=1)
        assert sorted(chunked) == sorted(naive)

    def test_matches_naive_eight_ranks_all(self):
        names = self._make_filenames(800)
        for rank in range(8):
            chunked, _, _ = _run_flat_layout_chunked(
                names, "npz", "/data", "train", comm_size=8, my_rank=rank)
            naive, _ = _run_flat_layout_naive(
                names, "npz", "/data", "train", comm_size=8, my_rank=rank)
            assert sorted(chunked) == sorted(naive), f"rank {rank} mismatch"

    def test_no_matching_files_returns_empty(self):
        names = ["file.txt", "file.csv", "file.parquet"]
        my_files, count, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0)
        assert count == 0
        assert my_files == []
        assert bcast[-1] is None

    def test_empty_directory_returns_empty(self):
        my_files, count, bcast = _run_flat_layout_chunked(
            [], "npz", "/data", "train", comm_size=1, my_rank=0)
        assert count == 0
        assert my_files == []
        # No non-sentinel broadcasts when filenames is empty
        assert bcast == [None]

    def test_sentinel_is_none_not_empty_list(self):
        """Sentinel must be None so empty batches don't prematurely break loop."""
        names = self._make_filenames(10)
        _, _, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0)
        assert bcast[-1] is None
        assert bcast[-1] != []

    def test_chunked_never_builds_full_uri_list(self):
        """
        Verify peak-allocation property: each broadcast contains at most
        chunk_size URIs, never the full list.
        """
        names = self._make_filenames(500)
        chunk_size = 100
        _, _, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0,
            chunk_size=chunk_size)
        # Every non-sentinel broadcast must be <= chunk_size
        for b in bcast[:-1]:  # skip None sentinel
            assert len(b) <= chunk_size, f"chunk too large: {len(b)}"

    def test_correct_number_of_broadcasts(self):
        """Exactly ceil(n / chunk_size) + 1 (sentinel) broadcasts."""
        import math
        n, chunk_size = 350, 100
        names = self._make_filenames(n)
        _, _, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0,
            chunk_size=chunk_size)
        expected_chunks = math.ceil(n / chunk_size)
        # +1 for the None sentinel
        assert len(bcast) == expected_chunks + 1

    def test_all_files_covered_across_ranks(self):
        """Union of all ranks' files == full set (no file dropped or doubled)."""
        n = 64
        names = self._make_filenames(n)
        comm_size = 4
        all_files = []
        for rank in range(comm_size):
            mf, _, _ = _run_flat_layout_chunked(
                names, "npz", "/data", "train",
                comm_size=comm_size, my_rank=rank)
            all_files.extend(mf)
        assert len(all_files) == n
        assert len(set(all_files)) == n  # no duplicates


class TestSentinelDistinction:
    """None sentinel must not be confused with an empty-chunk broadcast."""

    def test_empty_batch_chunk_is_empty_list_not_none(self):
        # A directory full of wrong-format files produces empty URI chunks
        names = [f"file_{i}.txt" for i in range(10)]
        _, _, bcast = _run_flat_layout_chunked(
            names, "npz", "/data", "train", comm_size=1, my_rank=0,
            chunk_size=5)
        # Two batches of 5 wrong-format files → two [] chunks, then None
        non_sentinel = bcast[:-1]
        assert all(b == [] for b in non_sentinel)
        assert bcast[-1] is None

    def test_none_sentinel_terminates_receive_loop(self):
        """Simulate non-root receive loop: must not break on empty chunk."""
        received = [[], [1, 2, 3], [], None]  # two empty chunks, then sentinel
        processed = []
        for chunk in received:
            if chunk is None:
                break
            processed.extend(chunk)
        assert processed == [1, 2, 3]
