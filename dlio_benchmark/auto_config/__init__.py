"""
dlio_benchmark.auto_config
==========================
Automatically generate DLIO configuration files from dftracer workload traces.

Pipeline:
  1. Instrument real workload with dftracer (instrument.py)
  2. Parse .pfw Chrome-trace files (parser.py)
  3. Extract DLIO schema from trace events (extractor.py)
  4. Emit validated DLIO YAML config (generator.py)
  5. Compare original vs. DLIO replay traces (compare.py)

CLI:
  python -m dlio_benchmark.auto_config run --command "..." --data-roots /data --output config.yaml
  python -m dlio_benchmark.auto_config analyze --trace-dir ./traces --output config.yaml
  python -m dlio_benchmark.auto_config validate --config config.yaml
  python -m dlio_benchmark.auto_config compare --original-trace ./t1 --dlio-trace ./t2
"""

from .confidence import Confidence, ParameterEstimate
from .parser import TraceEvent, parse_traces, find_trace_files
from .extractor import DLIOSchema, SchemaExtractor
from .generator import generate_yaml
from .instrument import DFTracerInstrumenter, DLTrainingTracer
from .compare import compare_traces, ComparisonReport
from .cli import main

__all__ = [
    "Confidence", "ParameterEstimate",
    "TraceEvent", "parse_traces", "find_trace_files",
    "DLIOSchema", "SchemaExtractor",
    "generate_yaml",
    "DFTracerInstrumenter", "DLTrainingTracer",
    "compare_traces", "ComparisonReport",
    "main",
]
