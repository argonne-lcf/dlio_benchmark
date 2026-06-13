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
import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# I/O categories emitted by dftracer
POSIX_CATS = {"POSIX", "STDIO"}
APP_CATS   = {"APP"}


@dataclass
class TraceEvent:
    name: str              # "read", "write", "open", "close", "epoch", "batch", ...
    cat: str               # "POSIX", "STDIO", "APP"
    ts: float              # microseconds since trace start
    dur: float             # duration in microseconds
    pid: int
    tid: int
    filename: str | None = None   # resolved absolute path (from args.filename)
    size: int | None = None       # bytes (from args.size for read/write)
    extra: dict = field(default_factory=dict)  # remaining args fields

    @property
    def ts_sec(self) -> float:
        return self.ts / 1e6

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur

    @property
    def end_ts_sec(self) -> float:
        return self.end_ts / 1e6

    def is_io(self) -> bool:
        return self.cat in POSIX_CATS

    def is_app(self) -> bool:
        return self.cat in APP_CATS


def _parse_line(line: str) -> TraceEvent | None:
    line = line.strip().rstrip(",")
    if not line or line in ("[", "]"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Preload traces can emit bare integers (timestamps/sizes) as top-level values
    if not isinstance(obj, dict):
        return None
    # Skip metadata events (ph != "X")
    if obj.get("ph") != "X":
        return None

    args = obj.get("args", {})

    # dftracer v0.0.dev1 preload traces store hashed identifiers instead of
    # plaintext paths. Use fhash as a synthetic filename so fd-to-file
    # grouping still works (open64 and read events share the same fhash).
    filename = (
        args.get("filename")
        or args.get("path")
        or args.get("pathname")
        or (f"__fhash__{args['fhash']}" if "fhash" in args else None)
    )
    # Actual bytes read is in 'ret'; 'size' is the legacy field name.
    # 'count' is the requested buffer size (often much larger than ret).
    size = args.get("size") or args.get("ret")

    return TraceEvent(
        name=obj.get("name", ""),
        cat=obj.get("cat", ""),
        ts=float(obj.get("ts", 0)),
        dur=float(obj.get("dur", 0)),
        pid=int(obj.get("pid", 0)),
        tid=int(obj.get("tid", 0)),
        filename=filename,
        size=size,
        extra={k: v for k, v in args.items() if k not in ("filename", "path", "pathname", "fhash", "size", "ret")},
    )


def parse_traces(pfw_paths: list[Path], data_roots: list[str] | None = None) -> list[TraceEvent]:
    """Parse one or more .pfw trace files into a sorted list of TraceEvents.

    Files from multiple ranks are merged and sorted by timestamp.
    If data_roots is given, filename paths are checked for membership to
    identify training data files vs. system/checkpoint files.
    """
    events: list[TraceEvent] = []
    roots = [str(Path(r).resolve()) for r in (data_roots or [])]

    for path in pfw_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Trace file not found: {path}")
            continue
        size_kb = path.stat().st_size // 1024
        logger.info(f"Parsing {path} ({size_kb} KB)")
        # Support both plain .pfw and gzip-compressed .pfw.gz
        opener = gzip.open if path.suffix == ".gz" else open
        mode = "rt" if path.suffix == ".gz" else "r"
        with opener(path, mode, errors="replace") as fh:
            for line in fh:
                ev = _parse_line(line)
                if ev is None:
                    continue
                # Resolve real paths; leave synthetic __fhash__ filenames as-is
                if ev.filename and not ev.filename.startswith("__fhash__"):
                    try:
                        ev.filename = str(Path(ev.filename).resolve())
                    except Exception:
                        pass
                events.append(ev)

    events.sort(key=lambda e: e.ts)
    logger.info(f"Parsed {len(events)} events from {len(pfw_paths)} trace file(s)")
    return events


def find_trace_files(trace_dir: str, pattern: str = "*.pfw*") -> list[Path]:
    """Return all .pfw and .pfw.gz files under trace_dir, sorted by rank number.

    dftracer >= 0.0.dev1 writes gzip-compressed .pfw.gz files; older versions
    write plain .pfw. Both are supported by parse_traces().
    """
    d = Path(trace_dir)
    seen = set()
    paths = []
    for p in sorted(d.glob("*.pfw")) + sorted(d.glob("*.pfw.gz")):
        if p not in seen:
            seen.add(p)
            paths.append(p)
    if not paths:
        paths = sorted(d.glob("*.json"))
    return paths
