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
import io
import json
import statistics
import tempfile
from pathlib import Path

import pytest
import yaml

from dlio_benchmark.auto_config.parser import TraceEvent, parse_traces
from dlio_benchmark.auto_config.extractor import SchemaExtractor
from dlio_benchmark.auto_config.generator import generate_yaml
from dlio_benchmark.auto_config.compare import compare_traces, _compute_stats
from dlio_benchmark.auto_config.confidence import Confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(name, cat, ts, dur, pid=1, tid=1, filename=None, size=None):
    return TraceEvent(name=name, cat=cat, ts=ts, dur=dur, pid=pid, tid=tid,
                      filename=filename, size=size)


def _pfw_line(name, cat, ts, dur, pid=1, tid=1, filename=None, size=None):
    args = {}
    if filename:
        args["filename"] = filename
    if size is not None:
        args["size"] = size
    return json.dumps({"name": name, "cat": cat, "ph": "X", "ts": ts,
                       "dur": dur, "pid": pid, "tid": tid, "args": args})


def _write_pfw(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Parser unit test
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_basic(self, tmp_path):
        pfw = tmp_path / "trace-0-of-1.pfw"
        _write_pfw(pfw, [
            _pfw_line("open",  "POSIX", 1000, 5,  filename="/data/train/img_000.npy"),
            _pfw_line("read",  "POSIX", 1010, 50, filename="/data/train/img_000.npy", size=65536),
            _pfw_line("close", "POSIX", 1060, 3,  filename="/data/train/img_000.npy"),
        ])
        events = parse_traces([pfw])
        assert len(events) == 3
        assert events[0].name == "open"
        assert events[1].size == 65536
        assert events[2].name == "close"

    def test_skip_metadata_events(self, tmp_path):
        pfw = tmp_path / "trace.pfw"
        _write_pfw(pfw, [
            '{"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "worker"}}',
            _pfw_line("read", "POSIX", 100, 10, size=4096),
        ])
        events = parse_traces([pfw])
        assert len(events) == 1
        assert events[0].name == "read"

    def test_sorted_by_timestamp(self, tmp_path):
        pfw = tmp_path / "trace.pfw"
        _write_pfw(pfw, [
            _pfw_line("read", "POSIX", 500, 10, size=1024),
            _pfw_line("read", "POSIX", 100, 10, size=1024),
            _pfw_line("read", "POSIX", 300, 10, size=1024),
        ])
        events = parse_traces([pfw])
        ts = [e.ts for e in events]
        assert ts == sorted(ts)


# ---------------------------------------------------------------------------
# 2. Format detection
# ---------------------------------------------------------------------------

class TestFormatDetection:
    def _schema(self, ext: str, n: int = 10) -> object:
        events = [
            _make_event("open", "POSIX", i * 1000, 5, filename=f"/data/train/file_{i}{ext}")
            for i in range(n)
        ]
        return SchemaExtractor(data_roots=["/data"]).extract(events)

    def test_hdf5(self):
        assert self._schema(".h5").format.value == "hdf5"

    def test_npy(self):
        assert self._schema(".npy").format.value == "npy"

    def test_tfrecord(self):
        assert self._schema(".tfrecord").format.value == "tfrecord"

    def test_jpeg(self):
        assert self._schema(".jpg").format.value == "jpeg"

    def test_high_confidence_many_files(self):
        schema = self._schema(".npy", n=20)
        assert schema.format.confidence == Confidence.HIGH


# ---------------------------------------------------------------------------
# 3. Batch size inference
# ---------------------------------------------------------------------------

class TestBatchInference:
    def _make_batched_events(self, batch_size: int, n_batches: int,
                             read_gap_us: float = 100, compute_gap_us: float = 50_000):
        """Synthetic trace: reads within a batch are close together,
        followed by a large compute gap."""
        events = []
        t = 0.0
        fnames = [f"/data/train/file_{i}.npy" for i in range(100)]
        idx = 0
        for _ in range(n_batches):
            for _ in range(batch_size):
                events.append(_make_event("read", "POSIX", t, 50,
                                          filename=fnames[idx % len(fnames)],
                                          size=65536))
                t += read_gap_us
                idx += 1
            t += compute_gap_us  # simulate GPU compute
        return events

    def test_batch_size_4(self):
        events = self._make_batched_events(batch_size=4, n_batches=20)
        schema = SchemaExtractor(data_roots=["/data"]).extract(events)
        assert schema.batch_size is not None
        assert schema.batch_size.value == 4

    def test_batch_size_32(self):
        events = self._make_batched_events(batch_size=32, n_batches=10)
        schema = SchemaExtractor(data_roots=["/data"]).extract(events)
        assert schema.batch_size is not None
        assert schema.batch_size.value == 32

    def test_read_threads(self):
        events = []
        t = 0.0
        for tid in range(4):
            for i in range(10):
                events.append(_make_event("read", "POSIX", t + i * 100, 50,
                                          tid=tid, filename=f"/data/f{i}.npy", size=65536))
        schema = SchemaExtractor(data_roots=["/data"]).extract(events)
        assert schema.read_threads.value == 4


# ---------------------------------------------------------------------------
# 4. Computation time inference
# ---------------------------------------------------------------------------

class TestComputationTime:
    def test_computation_time_mean(self):
        batch_size = 8
        compute_gap_us = 100_000  # 0.1 s
        events = []
        t = 0.0
        for _ in range(20):  # 20 batches
            for _ in range(batch_size):
                events.append(_make_event("read", "POSIX", t, 50,
                                          filename="/data/f.npy", size=65536))
                t += 200
            t += compute_gap_us

        schema = SchemaExtractor(data_roots=["/data"]).extract(events)
        assert schema.computation_time_mean is not None
        # Should be approximately 0.1 s ± some tolerance
        assert abs(schema.computation_time_mean.value - 0.1) < 0.02


# ---------------------------------------------------------------------------
# 5. Generator round-trip
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_round_trip(self, tmp_path):
        from dlio_benchmark.auto_config.extractor import DLIOSchema
        from dlio_benchmark.auto_config.confidence import ParameterEstimate, Confidence

        schema = DLIOSchema(
            format=ParameterEstimate("npy", Confidence.HIGH, "test"),
            data_folder=ParameterEstimate("/data/train", Confidence.HIGH, "test"),
            record_length=ParameterEstimate(65536, Confidence.HIGH, "test"),
            record_length_stdev=ParameterEstimate(0, Confidence.HIGH, "test"),
            num_files_train=ParameterEstimate(100, Confidence.HIGH, "test"),
            num_samples_per_file=ParameterEstimate(10, Confidence.MEDIUM, "test"),
            batch_size=ParameterEstimate(8, Confidence.HIGH, "test"),
            read_threads=ParameterEstimate(4, Confidence.HIGH, "test"),
            framework=ParameterEstimate("pytorch", Confidence.MEDIUM, "test"),
            computation_time_mean=ParameterEstimate(0.042, Confidence.HIGH, "test"),
            computation_time_stdev=ParameterEstimate(0.003, Confidence.HIGH, "test"),
            epochs=ParameterEstimate(1, Confidence.LOW, "test"),
        )
        out = tmp_path / "test_config.yaml"
        generate_yaml(schema, str(out), workload_name="test_workload")

        assert out.exists()
        with open(out) as f:
            content = f.read()
        # Parse ignoring comments
        clean = "\n".join(l.split("#")[0].rstrip() for l in content.splitlines())
        cfg = yaml.safe_load(clean)

        assert cfg["dataset"]["format"] == "npy"
        assert cfg["dataset"]["num_files_train"] == 100
        assert cfg["dataset"]["record_length"] == 65536
        assert cfg["reader"]["batch_size"] == 8
        assert cfg["reader"]["read_threads"] == 4

    def test_low_confidence_comment_present(self, tmp_path):
        from dlio_benchmark.auto_config.extractor import DLIOSchema
        from dlio_benchmark.auto_config.confidence import ParameterEstimate, Confidence

        schema = DLIOSchema(
            format=ParameterEstimate("npy", Confidence.HIGH, "test"),
            num_files_train=ParameterEstimate(5, Confidence.LOW, "test"),
            record_length=ParameterEstimate(1024, Confidence.LOW, "test"),
        )
        out = tmp_path / "low_conf.yaml"
        generate_yaml(schema, str(out))
        content = out.read_text()
        assert "LOW confidence" in content


# ---------------------------------------------------------------------------
# 6. Trace comparison
# ---------------------------------------------------------------------------

class TestCompare:
    def _identical_events(self, n_reads=100, size=65536):
        events = []
        t = 0.0
        for i in range(n_reads):
            events.append(_make_event("read", "POSIX", t, 50,
                                      filename=f"/data/f{i % 10}.npy", size=size))
            t += 1000
        return events

    def test_identical_traces_high_score(self):
        events = self._identical_events()
        report = compare_traces(events, events)
        assert report.overall_score > 0.9

    def test_different_read_sizes_lower_score(self):
        orig = self._identical_events(size=65536)
        dlio = self._identical_events(size=4096)
        report = compare_traces(orig, dlio)
        assert report.read_size_ks > 0.5
        assert report.overall_score < 0.9
