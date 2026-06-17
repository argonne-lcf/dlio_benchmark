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
Two modes of instrumentation:

1. **CLI mode** (DFTracerInstrumenter.run): wraps an arbitrary shell command
   with dftracer environment variables and LD_PRELOAD. Produces .pfw files.

2. **Python API mode** (DLTrainingTracer context manager): imported directly
   into a training script to emit epoch/batch application-level markers into
   the dftracer trace, enabling precise batch-size and computation-time
   extraction.

Usage (CLI mode):
    instrumenter = DFTracerInstrumenter(data_roots=["/data/imagenet"], output_dir="./traces")
    pfw_files = instrumenter.run("python train.py --epochs 1")

Usage (in training script):
    from dlio_benchmark.auto_config.instrument import DLTrainingTracer
    tracer = DLTrainingTracer(output_dir="./traces", data_roots=["/data/imagenet"])
    tracer.initialize()
    for epoch in range(num_epochs):
        with tracer.epoch(epoch):
            for batch_idx, batch in enumerate(dataloader):
                with tracer.batch(epoch, batch_idx):
                    # ... training step ...
                    pass
    tracer.finalize()
"""

import builtins
import io
import json
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_DFTRACER_ENABLE = "DFTRACER_ENABLE"
_DFTRACER_LOG_FILE = "DFTRACER_LOG_FILE"
_DFTRACER_DATA_DIR = "DFTRACER_DATA_DIR"
_DFTRACER_INC_METADATA = "DFTRACER_INC_METADATA"
_DFTRACER_INIT = "DFTRACER_INIT"


class DFTracerInstrumenter:
    """Wrap an arbitrary shell command with dftracer POSIX-level instrumentation."""

    def __init__(self, data_roots: list[str], output_dir: str):
        self.data_roots = [str(Path(r).resolve()) for r in data_roots]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_prefix = str(self.output_dir / "trace")

    def _env(self) -> dict:
        env = os.environ.copy()
        env[_DFTRACER_ENABLE] = "1"
        env[_DFTRACER_LOG_FILE] = self.trace_prefix
        env[_DFTRACER_DATA_DIR] = ":".join(self.data_roots)
        env[_DFTRACER_INC_METADATA] = "1"
        # dftracer 2.x: DFTRACER_INIT=0 disables auto-init so the Python API
        # or LD_PRELOAD handles it; set to 1 for LD_PRELOAD-only mode
        env[_DFTRACER_INIT] = "1"
        return env

    def run(self, command: str) -> list[Path]:
        """Run command under dftracer and return collected .pfw file paths."""
        logger.info(f"Running instrumented command: {command}")
        logger.info(f"Trace output dir: {self.output_dir}")
        logger.info(f"Data roots: {self.data_roots}")

        env = self._env()
        try:
            result = subprocess.run(
                command, shell=True, env=env,
                stdout=None, stderr=None  # inherit parent streams
            )
            if result.returncode != 0:
                logger.warning(f"Command exited with code {result.returncode}")
        except KeyboardInterrupt:
            logger.info("Interrupted — collecting traces so far")

        pfw_files = sorted(self.output_dir.glob("*.pfw"))
        if not pfw_files:
            pfw_files = sorted(self.output_dir.glob("trace*.json"))
        logger.info(f"Found {len(pfw_files)} trace file(s)")
        return pfw_files


class PythonIOTracer:
    """Pure-Python fallback tracer for macOS / environments without the native dftracer backend.

    Monkey-patches builtins.open to record every file open/read/close event
    for paths under data_roots, and writes Chrome trace JSON Lines (.pfw) output.
    Thread-safe; uses perf_counter_ns for microsecond timestamps.
    """

    def __init__(self, output_dir: str, data_roots: list[str] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_roots = [str(Path(r).resolve()) for r in (data_roots or [])]
        self._pfw_path = self.output_dir / "trace-0-of-1.pfw"
        self._lock = threading.Lock()
        self._events: list[str] = []
        self._original_open = builtins.open
        self._active = False
        self._t0_ns = 0

    def _ts_us(self) -> float:
        return (time.perf_counter_ns() - self._t0_ns) / 1_000

    def _in_roots(self, path: str) -> bool:
        if not self.data_roots:
            return True
        try:
            p = str(Path(path).resolve())
            return any(p.startswith(r) for r in self.data_roots)
        except Exception:
            return False

    def _emit(self, name: str, cat: str, ts: float, dur: float,
              filename: str | None = None, size: int | None = None) -> None:
        args: dict = {}
        if filename:
            args["filename"] = filename
        if size is not None:
            args["size"] = size
        ev = {"name": name, "cat": cat, "ph": "X",
              "ts": round(ts, 3), "dur": max(round(dur, 3), 0.001),
              "pid": os.getpid(), "tid": threading.get_ident(), "args": args}
        line = json.dumps(ev)
        with self._lock:
            self._events.append(line)

    def _make_wrapped_open(self):
        tracer = self

        def patched_open(file, mode="r", *args, **kwargs):
            fh = tracer._original_open(file, mode, *args, **kwargs)
            if isinstance(file, (str, bytes, Path)) and "r" in mode:
                path = str(file)
                if tracer._active and tracer._in_roots(path):
                    ts = tracer._ts_us()
                    tracer._emit("open", "POSIX", ts, 0.1, filename=path)
                    return _TracedFile(fh, path, tracer)
            return fh

        return patched_open

    def start(self) -> None:
        self._t0_ns = time.perf_counter_ns()
        self._active = True
        builtins.open = self._make_wrapped_open()
        logger.info(f"PythonIOTracer started → {self._pfw_path}")

    def stop(self) -> None:
        self._active = False
        builtins.open = self._original_open
        self._pfw_path.write_text("\n".join(self._events) + "\n")
        logger.info(f"PythonIOTracer: wrote {len(self._events)} events to {self._pfw_path}")

    def emit_app(self, name: str, ts: float, dur: float, **kwargs) -> None:
        args = {k: v for k, v in kwargs.items()}
        ev = {"name": name, "cat": "APP", "ph": "X",
              "ts": round(ts, 3), "dur": max(round(dur, 3), 0.001),
              "pid": os.getpid(), "tid": threading.get_ident(), "args": args}
        with self._lock:
            self._events.append(json.dumps(ev))


class _TracedFile:
    """Thin wrapper around a file object that records read() calls."""

    def __init__(self, fh: io.IOBase, path: str, tracer: "PythonIOTracer"):
        self._fh = fh
        self._path = path
        self._tracer = tracer

    def read(self, size=-1):
        ts = self._tracer._ts_us()
        data = self._fh.read(size)
        dur = self._tracer._ts_us() - ts
        if data:
            self._tracer._emit("read", "POSIX", ts, dur,
                               filename=self._path, size=len(data))
        return data

    def readline(self, size=-1):
        ts = self._tracer._ts_us()
        data = self._fh.readline(size)
        dur = self._tracer._ts_us() - ts
        if data:
            self._tracer._emit("read", "POSIX", ts, dur,
                               filename=self._path, size=len(data))
        return data

    def readlines(self, hint=-1):
        ts = self._tracer._ts_us()
        lines = self._fh.readlines(hint)
        dur = self._tracer._ts_us() - ts
        total = sum(len(l) for l in lines)
        if total:
            self._tracer._emit("read", "POSIX", ts, dur,
                               filename=self._path, size=total)
        return lines

    def close(self):
        ts = self._tracer._ts_us()
        self._fh.close()
        self._tracer._emit("close", "POSIX", ts, 0.1, filename=self._path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return iter(self._fh)

    def __getattr__(self, name):
        return getattr(self._fh, name)


class DLTrainingTracer:
    """Python API tracer for DL training loops.

    Uses the native dftracer backend when available (Linux); falls back to
    PythonIOTracer on macOS or environments without the native C++ backend.
    Emits epoch/batch application-level markers for precise batch-size and
    computation-time extraction.
    """

    def __init__(self, output_dir: str, data_roots: list[str] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_roots = [str(Path(r).resolve()) for r in (data_roots or [])]
        self._tracer = None
        self._Profile = None
        self._py_tracer: PythonIOTracer | None = None

    def _native_available(self) -> bool:
        try:
            from dftracer.python.common import profiler, NoOpProfiler
            # dftracer >= 0.0.dev1: native backend type is 'module' (C extension);
            # older pydftracer-based builds: not isinstance(profiler, NoOpProfiler)
            if isinstance(profiler, NoOpProfiler):
                return False
            # module type = native C++ backend loaded successfully
            return type(profiler).__name__ != "NoOpProfiler"
        except Exception:
            return False

    def emit_config(self, batch_size: int, num_workers: int = 0,
                    num_samples: int = 0, **kwargs) -> None:
        """Emit training configuration as a zero-duration APP event.

        Call this after initialize() and before the first epoch so that the
        extractor can read batch_size directly from the trace rather than
        approximating it from POSIX read counts.

        Args:
            batch_size:   DataLoader batch size.
            num_workers:  DataLoader num_workers.
            num_samples:  Total dataset samples (optional).
        """
        if self._Profile is not None:
            # dftracer Profile only accepts epoch/step/image_idx/image_size;
            # emit a zero-duration APP event with metadata encoded in image_size.
            # We use a dedicated 'dlio_config' name so the extractor can find it.
            try:
                from dftracer.python import dft_fn as Profile
                with Profile(cat="APP", name="dlio_config",
                             image_size=batch_size, step=num_workers):
                    pass
            except Exception:
                pass
        elif self._py_tracer is not None:
            ts = self._py_tracer._ts_us()
            self._py_tracer.emit_app("dlio_config", ts, 0.001,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     num_samples=num_samples,
                                     **kwargs)

    def initialize(self) -> None:
        """Initialize tracing. Call once before the training loop starts."""
        if self._native_available():
            from dftracer.python import dftracer as PerfTrace, dft_fn as Profile
            self._Profile = Profile
            trace_file = str(self.output_dir / "trace")
            data_dir = ":".join(self.data_roots) if self.data_roots else "."
            self._tracer = PerfTrace.initialize_log(
                logfile=trace_file, data_dir=data_dir, process_id=-1
            )
            logger.info(f"DFTracer (native) initialized → {trace_file}-*.pfw")
        else:
            logger.info("Native dftracer backend not available; using PythonIOTracer fallback")
            self._py_tracer = PythonIOTracer(
                output_dir=str(self.output_dir), data_roots=self.data_roots
            )
            self._py_tracer.start()

    def finalize(self) -> None:
        """Flush and close the trace. Call once after training ends."""
        if self._tracer is not None:
            try:
                self._tracer.finalize()
                logger.info("DFTracer finalized")
            except Exception as e:
                logger.warning(f"DFTracer finalize error: {e}")
        if self._py_tracer is not None:
            self._py_tracer.stop()

    @contextmanager
    def epoch(self, epoch_num: int):
        """Emit an APP-level 'epoch' span."""
        if self._Profile is not None:
            with self._Profile(cat="APP", name=f"epoch_{epoch_num}"):
                yield
        elif self._py_tracer is not None:
            ts = self._py_tracer._ts_us()
            yield
            self._py_tracer.emit_app(f"epoch_{epoch_num}", ts,
                                     self._py_tracer._ts_us() - ts, epoch=epoch_num)
        else:
            yield

    @contextmanager
    def batch(self, epoch_num: int, batch_idx: int):
        """Emit an APP-level 'batch' span."""
        if self._Profile is not None:
            with self._Profile(cat="APP", name="batch", epoch=epoch_num, step=batch_idx):
                yield
        elif self._py_tracer is not None:
            ts = self._py_tracer._ts_us()
            yield
            self._py_tracer.emit_app("batch", ts,
                                     self._py_tracer._ts_us() - ts,
                                     epoch=epoch_num, step=batch_idx)
        else:
            yield

    @contextmanager
    def checkpoint(self, epoch_num: int):
        """Emit an APP-level 'checkpoint' span."""
        if self._Profile is not None:
            with self._Profile(cat="APP", name="checkpoint", epoch=epoch_num):
                yield
        elif self._py_tracer is not None:
            ts = self._py_tracer._ts_us()
            yield
            self._py_tracer.emit_app("checkpoint", ts,
                                     self._py_tracer._ts_us() - ts, epoch=epoch_num)
        else:
            yield
