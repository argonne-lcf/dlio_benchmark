"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
"""
Trace similarity comparison between an original workload trace and a DLIO
replay trace. Reports per-metric similarity scores and an overall score.

Metrics compared:
  1. Read throughput (MB/s)           — how fast data is read
  2. Read size distribution           — histogram of read() call sizes (KS distance)
  3. Inter-burst gap distribution     — computation time pattern (KS distance)
  4. I/O concurrency (thread count)   — number of reader threads
  5. Burst size (batch size)          — samples per I/O burst
  6. File access pattern              — sequential vs. random file order
"""

import logging
import math
import statistics
from dataclasses import dataclass
from collections import Counter

from .parser import TraceEvent

logger = logging.getLogger(__name__)

_MIN_READ = 512  # bytes — filter out metadata reads


@dataclass
class TraceStats:
    """Summarises I/O behaviour from a parsed trace."""
    total_bytes_read: int = 0
    wall_time_sec: float = 0.0
    throughput_mbs: float = 0.0       # MB/s
    read_sizes: list[int] = None      # individual read() sizes
    inter_burst_gaps_sec: list[float] = None  # compute gap durations
    burst_sizes: list[int] = None     # reads per burst
    num_threads: int = 0
    num_files: int = 0

    def __post_init__(self):
        if self.read_sizes is None:
            self.read_sizes = []
        if self.inter_burst_gaps_sec is None:
            self.inter_burst_gaps_sec = []
        if self.burst_sizes is None:
            self.burst_sizes = []


@dataclass
class ComparisonReport:
    throughput_ratio: float            # DLIO / original (1.0 = perfect)
    read_size_ks: float                # KS distance [0,1] — lower is better
    burst_gap_ks: float                # KS distance [0,1]
    thread_ratio: float                # DLIO / original
    burst_size_ratio: float            # DLIO / original
    overall_score: float               # weighted average similarity [0,1]

    def summary(self) -> str:
        lines = [
            "=== Trace Similarity Report ===",
            f"  Throughput ratio (DLIO/orig):  {self.throughput_ratio:.2f}  "
            f"({'OK' if 0.8 <= self.throughput_ratio <= 1.2 else 'MISMATCH'})",
            f"  Read-size KS distance:         {self.read_size_ks:.3f}  "
            f"({'OK' if self.read_size_ks < 0.1 else 'MISMATCH'})",
            f"  Burst-gap KS distance:         {self.burst_gap_ks:.3f}  "
            f"({'OK' if self.burst_gap_ks < 0.15 else 'MISMATCH'})",
            f"  Thread ratio (DLIO/orig):      {self.thread_ratio:.2f}",
            f"  Burst-size ratio (DLIO/orig):  {self.burst_size_ratio:.2f}",
            f"  Overall similarity score:      {self.overall_score:.2f} / 1.00",
        ]
        return "\n".join(lines)


def _compute_stats(events: list[TraceEvent]) -> TraceStats:
    reads = [e for e in events if e.name in ("read", "pread64", "pread")
             and e.size and e.size >= _MIN_READ]
    if not reads:
        return TraceStats()

    stats = TraceStats()
    stats.total_bytes_read = sum(e.size for e in reads)
    stats.num_threads = len({e.tid for e in reads})
    stats.num_files = len({e.filename for e in reads if e.filename})
    stats.read_sizes = [e.size for e in reads]

    ts_sorted = sorted(e.ts for e in reads)
    if ts_sorted:
        wall_us = ts_sorted[-1] - ts_sorted[0]
        stats.wall_time_sec = wall_us / 1e6
        stats.throughput_mbs = (stats.total_bytes_read / (1024**2)) / max(stats.wall_time_sec, 1e-9)

    # Burst detection (same logic as extractor)
    gaps_us = [ts_sorted[i+1] - ts_sorted[i] for i in range(len(ts_sorted)-1)]
    if gaps_us:
        median_gap = statistics.median(gaps_us)
        threshold = median_gap * 10
        current = 1
        for g in gaps_us:
            if g > threshold:
                stats.burst_sizes.append(current)
                stats.inter_burst_gaps_sec.append(g / 1e6)
                current = 1
            else:
                current += 1
        stats.burst_sizes.append(current)

    return stats


def _ks_distance(a: list[float], b: list[float]) -> float:
    """Kolmogorov-Smirnov distance between two empirical distributions. Returns [0, 1]."""
    if not a and not b:
        return 0.0  # both empty → identical
    if not a or not b:
        return 1.0
    all_vals = sorted(set(a + b))
    n_a, n_b = len(a), len(b)
    ca = Counter(a)
    cb = Counter(b)
    cum_a = cum_b = 0.0
    max_diff = 0.0
    for v in all_vals:
        cum_a += ca.get(v, 0) / n_a
        cum_b += cb.get(v, 0) / n_b
        max_diff = max(max_diff, abs(cum_a - cum_b))
    return max_diff


def _ratio_score(ratio: float) -> float:
    """Convert a ratio to a [0,1] similarity score. 1.0 = perfect match."""
    return max(0.0, 1.0 - abs(ratio - 1.0))


def compare_traces(
    original_events: list[TraceEvent],
    dlio_events: list[TraceEvent],
) -> ComparisonReport:
    """Compare original workload trace vs DLIO replay trace.

    Returns a ComparisonReport with per-metric and overall similarity scores.
    """
    orig = _compute_stats(original_events)
    dlio = _compute_stats(dlio_events)

    # 1. Throughput ratio
    if orig.throughput_mbs > 0:
        tp_ratio = dlio.throughput_mbs / orig.throughput_mbs
    else:
        tp_ratio = 1.0

    # 2. Read size KS distance
    rs_ks = _ks_distance(
        [float(x) for x in orig.read_sizes],
        [float(x) for x in dlio.read_sizes]
    )

    # 3. Inter-burst gap KS distance
    gap_ks = _ks_distance(orig.inter_burst_gaps_sec, dlio.inter_burst_gaps_sec)

    # 4. Thread ratio
    thread_ratio = dlio.num_threads / max(orig.num_threads, 1)

    # 5. Burst size ratio
    orig_burst = statistics.median(orig.burst_sizes) if orig.burst_sizes else 1
    dlio_burst  = statistics.median(dlio.burst_sizes)  if dlio.burst_sizes  else 1
    burst_ratio = dlio_burst / max(orig_burst, 1)

    # Overall score: weighted average of per-metric similarities
    weights = {
        "throughput": 0.30,
        "read_size":  0.25,
        "burst_gap":  0.25,
        "threads":    0.10,
        "burst_size": 0.10,
    }
    scores = {
        "throughput": _ratio_score(tp_ratio),
        "read_size":  1.0 - rs_ks,
        "burst_gap":  1.0 - gap_ks,
        "threads":    _ratio_score(thread_ratio),
        "burst_size": _ratio_score(burst_ratio),
    }
    overall = sum(scores[k] * w for k, w in weights.items())

    logger.info(f"Per-metric scores: {scores}")
    logger.info(f"Overall similarity: {overall:.3f}")

    return ComparisonReport(
        throughput_ratio=tp_ratio,
        read_size_ks=rs_ks,
        burst_gap_ks=gap_ks,
        thread_ratio=thread_ratio,
        burst_size_ratio=burst_ratio,
        overall_score=overall,
    )
