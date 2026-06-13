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
import logging
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .confidence import Confidence, ParameterEstimate
from .parser import TraceEvent

logger = logging.getLogger(__name__)

# File extension → DLIO format string
_EXT_MAP = {
    ".h5": "hdf5", ".hdf5": "hdf5",
    ".npy": "npy",
    ".npz": "npz",
    ".tfrecord": "tfrecord",
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".csv": "csv",
    ".bin": "indexed_binary",
}

# Minimum read size to be considered a sample read (filters out metadata reads).
# Small image formats (PNG/JPEG) can be <512 bytes; use 32 bytes as the floor.
_MIN_SAMPLE_READ = 32  # bytes


@dataclass
class DLIOSchema:
    # Dataset
    format: ParameterEstimate[str] | None = None
    data_folder: ParameterEstimate[str] | None = None
    record_length: ParameterEstimate[int] | None = None
    record_length_stdev: ParameterEstimate[int] | None = None
    num_files_train: ParameterEstimate[int] | None = None
    num_samples_per_file: ParameterEstimate[int] | None = None
    # Reader
    batch_size: ParameterEstimate[int] | None = None
    read_threads: ParameterEstimate[int] | None = None
    prefetch_size: ParameterEstimate[int] | None = None
    file_shuffle: ParameterEstimate[str] | None = None
    sample_shuffle: ParameterEstimate[str] | None = None
    # Training
    framework: ParameterEstimate[str] | None = None
    computation_time_mean: ParameterEstimate[float] | None = None
    computation_time_stdev: ParameterEstimate[float] | None = None
    epochs: ParameterEstimate[int] | None = None
    # Checkpointing
    do_checkpoint: ParameterEstimate[bool] | None = None
    checkpoint_interval_epochs: ParameterEstimate[int] | None = None


def _confidence_from_count(n: int, cv: float = 0.0) -> Confidence:
    """Assign confidence based on observation count and coefficient of variation."""
    if n >= 100 and cv < 0.2:
        return Confidence.HIGH
    if n >= 10 and cv < 0.5:
        return Confidence.MEDIUM
    return Confidence.LOW


def _mode(values: list) -> tuple:
    """Return (mode_value, fraction_that_match)."""
    if not values:
        return None, 0.0
    c = Counter(values)
    mode_val, count = c.most_common(1)[0]
    return mode_val, count / len(values)


class SchemaExtractor:
    def __init__(self, data_roots: list[str] | None = None):
        # Store both the raw roots and their realpath equivalents so we match
        # Lustre paths (/lus/eagle/...) and NFS aliases (/eagle/...) equally.
        raw = [str(Path(r).resolve()) for r in (data_roots or [])]
        real = [os.path.realpath(r) for r in raw]
        self.data_roots = list(dict.fromkeys(raw + real))  # dedup, order-preserving

    def _in_data_roots(self, filename: str | None) -> bool:
        if not filename or not self.data_roots:
            return True  # no filter → accept everything
        # Synthetic fhash filenames (from preload traces) always pass through
        if filename.startswith("__fhash__"):
            return True
        # Preload traces may record different path aliases for the same filesystem
        fn_real = os.path.realpath(filename) if filename else filename
        return any(
            filename.startswith(root) or fn_real.startswith(root)
            for root in self.data_roots
        )

    def extract(self, events: list[TraceEvent]) -> DLIOSchema:
        schema = DLIOSchema()

        # POSIX preload traces: read() uses file descriptors, not paths, so
        # filename is often None. Accept all read events with a size — they are
        # all data reads during training.
        read_events = [
            e for e in events
            if e.name in ("read", "pread64", "pread") and e.is_io()
            and e.size and e.size >= _MIN_SAMPLE_READ
            and (e.filename is None or self._in_data_roots(e.filename))
        ]
        open_events = [
            e for e in events
            if e.name in ("open", "open64", "openat") and e.is_io()
            and self._in_data_roots(e.filename)
        ]
        write_events = [
            e for e in events
            if e.name in ("write", "pwrite64", "pwrite") and e.is_io()
            and not self._in_data_roots(e.filename)
        ]
        app_events = [e for e in events if e.is_app()]

        self._extract_format(schema, open_events)
        self._extract_file_inventory(schema, open_events)
        self._extract_record_length(schema, read_events)
        self._extract_samples_per_file(schema, read_events, open_events)
        self._extract_batch_and_threads(schema, read_events)
        self._extract_computation_time(schema, read_events)
        self._extract_shuffle(schema, open_events, read_events)
        self._extract_framework(schema, open_events, app_events)
        self._extract_epochs(schema, app_events, open_events)
        self._extract_checkpoint(schema, write_events, app_events)

        return schema

    # ------------------------------------------------------------------
    def _extract_format(self, schema: DLIOSchema, opens: list[TraceEvent]) -> None:
        exts = []
        for ev in opens:
            if ev.filename:
                ext = Path(ev.filename).suffix.lower()
                if ext in _EXT_MAP:
                    exts.append(ext)
        if not exts:
            return
        mode_ext, frac = _mode(exts)
        fmt = _EXT_MAP[mode_ext]
        conf = Confidence.HIGH if frac > 0.9 and len(exts) >= 5 else Confidence.MEDIUM
        schema.format = ParameterEstimate(
            value=fmt, confidence=conf,
            source=f"file extension '{mode_ext}' in {frac*100:.0f}% of {len(exts)} opens"
        )
        # Set data_folder as common prefix of opened data files
        data_files = [ev.filename for ev in opens if ev.filename and Path(ev.filename).suffix.lower() in _EXT_MAP]
        if data_files:
            common = str(Path(data_files[0]).parent)
            schema.data_folder = ParameterEstimate(
                value=common, confidence=Confidence.MEDIUM,
                source="common parent of opened data files"
            )

    def _extract_file_inventory(self, schema: DLIOSchema, opens: list[TraceEvent]) -> None:
        data_files = set()
        for ev in opens:
            if ev.filename and Path(ev.filename).suffix.lower() in _EXT_MAP:
                data_files.add(ev.filename)
        n = len(data_files)
        if n == 0:
            return
        conf = _confidence_from_count(n)
        schema.num_files_train = ParameterEstimate(
            value=n, confidence=conf,
            source=f"{n} unique data files opened"
        )

    def _extract_record_length(self, schema: DLIOSchema, reads: list[TraceEvent]) -> None:
        sizes = [e.size for e in reads if e.size]
        if not sizes:
            return
        mode_size, frac = _mode(sizes)
        mean = statistics.mean(sizes)
        stdev = statistics.stdev(sizes) if len(sizes) > 1 else 0
        cv = stdev / mean if mean > 0 else 0
        conf = _confidence_from_count(len(sizes), cv)
        schema.record_length = ParameterEstimate(
            value=int(mode_size), confidence=conf,
            source=f"mode of {len(sizes)} read sizes ({frac*100:.0f}% match)"
        )
        schema.record_length_stdev = ParameterEstimate(
            value=int(stdev), confidence=conf,
            source=f"stdev of read sizes (cv={cv:.2f})"
        )

    def _extract_samples_per_file(self, schema: DLIOSchema,
                                   reads: list[TraceEvent],
                                   opens: list[TraceEvent] | None = None) -> None:
        # Primary path: count reads per file when filenames are available
        reads_per_file: dict[str, int] = defaultdict(int)
        for ev in reads:
            if ev.filename:
                reads_per_file[ev.filename] += 1

        if reads_per_file:
            counts = list(reads_per_file.values())
            median_count = int(statistics.median(counts))
            n = len(counts)
            schema.num_samples_per_file = ParameterEstimate(
                value=median_count, confidence=_confidence_from_count(n),
                source=f"median reads per file across {n} files"
            )
            return

        # Fallback for preload traces (read() has no filename): estimate from
        # total reads ÷ unique files opened in data_roots.
        if opens and schema.num_files_train and len(reads) > 0:
            n_files = schema.num_files_train.value
            if n_files > 0:
                # total reads / files_seen gives reads-per-file per epoch
                n_epochs = schema.epochs.value if schema.epochs else 1
                per_file = max(1, round(len(reads) / max(n_files, 1) / n_epochs))
                schema.num_samples_per_file = ParameterEstimate(
                    value=per_file,
                    confidence=Confidence.LOW,
                    source=f"estimated: {len(reads)} reads / {n_files} files / {n_epochs} epochs"
                )

    def _extract_batch_and_threads(self, schema: DLIOSchema, reads: list[TraceEvent]) -> None:
        if len(reads) < 2:
            return

        # Group reads by TID to count threads
        tids = {e.tid for e in reads}
        schema.read_threads = ParameterEstimate(
            value=len(tids),
            confidence=Confidence.HIGH if len(reads) > 20 else Confidence.MEDIUM,
            source=f"{len(tids)} distinct TIDs issued read calls"
        )

        # Batch inference: find gaps between read bursts
        timestamps = sorted(e.ts for e in reads)
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if not gaps:
            return
        median_gap = statistics.median(gaps)

        # A "burst boundary" is a gap > 10x the median inter-read gap
        burst_threshold = median_gap * 10
        burst_sizes = []
        current = 1
        for g in gaps:
            if g > burst_threshold:
                burst_sizes.append(current)
                current = 1
            else:
                current += 1
        burst_sizes.append(current)

        if not burst_sizes:
            return
        batch_mode, frac = _mode(burst_sizes)
        n = len(burst_sizes)
        conf = _confidence_from_count(n, 0.0 if frac > 0.8 else 0.5)
        schema.batch_size = ParameterEstimate(
            value=int(batch_mode), confidence=conf,
            source=f"modal burst size={batch_mode} ({frac*100:.0f}% of {n} bursts)"
        )

        # Prefetch: detect overlap between I/O bursts and compute gaps
        # If reads start before the previous burst's compute gap ends, prefetch is active
        if n >= 3:
            schema.prefetch_size = ParameterEstimate(
                value=2,  # default conservative estimate
                confidence=Confidence.LOW,
                source="default estimate; overlap analysis requires epoch markers"
            )

    def _extract_computation_time(self, schema: DLIOSchema, reads: list[TraceEvent]) -> None:
        if not schema.batch_size or len(reads) < 4:
            return
        batch_size = schema.batch_size.value
        if batch_size == 0:
            return

        timestamps = sorted(e.ts for e in reads)
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        median_gap = statistics.median(gaps)
        burst_threshold = median_gap * 10

        # Collect inter-burst gaps (= computation time)
        compute_gaps_us = []
        i = 0
        in_burst = 0
        for j, g in enumerate(gaps):
            in_burst += 1
            if g > burst_threshold:
                compute_gaps_us.append(g)
                in_burst = 0

        if len(compute_gaps_us) < 2:
            return
        mean_us = statistics.mean(compute_gaps_us)
        stdev_us = statistics.stdev(compute_gaps_us) if len(compute_gaps_us) > 1 else 0
        cv = stdev_us / mean_us if mean_us > 0 else 0
        conf = _confidence_from_count(len(compute_gaps_us), cv)
        schema.computation_time_mean = ParameterEstimate(
            value=round(mean_us / 1e6, 4), confidence=conf,
            source=f"mean of {len(compute_gaps_us)} inter-burst gaps"
        )
        schema.computation_time_stdev = ParameterEstimate(
            value=round(stdev_us / 1e6, 4), confidence=conf,
            source=f"stdev of inter-burst gaps (cv={cv:.2f})"
        )

    def _extract_shuffle(self, schema: DLIOSchema, opens: list[TraceEvent], reads: list[TraceEvent]) -> None:
        # File shuffle: check if open order differs from sorted filename order
        opened_files = [ev.filename for ev in opens if ev.filename and Path(ev.filename).suffix.lower() in _EXT_MAP]
        if len(opened_files) >= 3:
            sorted_files = sorted(opened_files)
            diffs = sum(1 for a, b in zip(opened_files, sorted_files) if a != b)
            shuffled = diffs / len(opened_files) > 0.3
            schema.file_shuffle = ParameterEstimate(
                value="seed" if shuffled else "off",
                confidence=Confidence.MEDIUM,
                source=f"{diffs}/{len(opened_files)} files opened out of sorted order"
            )

        # Sample shuffle: check if read offsets within a single file are non-monotonic
        by_file: dict[str, list] = defaultdict(list)
        for ev in reads:
            if ev.filename:
                by_file[ev.filename].append(ev.ts)
        # Use read timestamps as proxy for offset ordering (dftracer may not expose offset)
        # Heuristic: if same file is read by multiple TIDs interleaved, sample shuffle likely
        multi_tid_files = sum(
            1 for f, ts_list in by_file.items()
            if len({e.tid for e in reads if e.filename == f}) > 1
        )
        if multi_tid_files > 0:
            schema.sample_shuffle = ParameterEstimate(
                value="seed",
                confidence=Confidence.LOW,
                source=f"{multi_tid_files} files read by multiple threads (shuffle inferred)"
            )
        else:
            schema.sample_shuffle = ParameterEstimate(
                value="off", confidence=Confidence.LOW,
                source="single-thread reads per file (shuffle status uncertain)"
            )

    def _extract_framework(self, schema: DLIOSchema, opens: list[TraceEvent], apps: list[TraceEvent]) -> None:
        # Check app-level event names
        app_names = " ".join(e.name for e in apps).lower()
        if "torch" in app_names or "pytorch" in app_names:
            schema.framework = ParameterEstimate("pytorch", Confidence.HIGH, "app event names")
            return
        if "tensorflow" in app_names or "tf." in app_names:
            schema.framework = ParameterEstimate("tensorflow", Confidence.HIGH, "app event names")
            return
        # Fall back to open filenames (site-packages paths)
        all_fnames = " ".join(ev.filename or "" for ev in opens).lower()
        if "torch" in all_fnames:
            schema.framework = ParameterEstimate("pytorch", Confidence.MEDIUM, "torch in opened paths")
        elif "tensorflow" in all_fnames:
            schema.framework = ParameterEstimate("tensorflow", Confidence.MEDIUM, "tensorflow in opened paths")
        else:
            schema.framework = ParameterEstimate("pytorch", Confidence.LOW, "default (not detected)")

    def _extract_epochs(self, schema: DLIOSchema, apps: list[TraceEvent], opens: list[TraceEvent]) -> None:
        epoch_events = [e for e in apps if "epoch" in e.name.lower()]
        if epoch_events:
            n_epochs = len({e.extra.get("epoch", i) for i, e in enumerate(epoch_events)})
            schema.epochs = ParameterEstimate(n_epochs, Confidence.HIGH, "epoch app markers")
        else:
            # Heuristic: count how many times the same file set is re-opened
            data_files = [ev.filename for ev in opens if ev.filename and Path(ev.filename).suffix.lower() in _EXT_MAP]
            if data_files:
                file_counts = Counter(data_files)
                max_repeats = max(file_counts.values())
                schema.epochs = ParameterEstimate(
                    max_repeats, Confidence.LOW,
                    f"max re-open count per file={max_repeats} (heuristic)"
                )

    def _extract_checkpoint(self, schema: DLIOSchema, writes: list[TraceEvent], apps: list[TraceEvent]) -> None:
        ckpt_app = [e for e in apps if "checkpoint" in e.name.lower() or "save" in e.name.lower()]
        if ckpt_app:
            schema.do_checkpoint = ParameterEstimate(True, Confidence.HIGH, "checkpoint app markers")
        elif writes:
            schema.do_checkpoint = ParameterEstimate(
                True, Confidence.MEDIUM,
                f"{len(writes)} write events outside data roots"
            )
        else:
            schema.do_checkpoint = ParameterEstimate(False, Confidence.MEDIUM, "no write events detected")
