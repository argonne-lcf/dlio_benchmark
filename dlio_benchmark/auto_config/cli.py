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
CLI entry point for dlio_benchmark.auto_config.

Sub-commands:
  run       -- instrument a workload and generate a DLIO config
  analyze   -- parse existing .pfw traces and generate a DLIO config
  validate  -- check that a DLIO config YAML is well-formed
  compare   -- compare an original trace against a DLIO replay trace

Examples:
  python -m dlio_benchmark.auto_config run \\
      --command "python train.py --epochs 1" \\
      --data-roots /data/imagenet \\
      --output dlio_imagenet.yaml

  python -m dlio_benchmark.auto_config analyze \\
      --trace-dir ./traces \\
      --data-roots /data/imagenet \\
      --output dlio_imagenet.yaml

  python -m dlio_benchmark.auto_config validate \\
      --config dlio_imagenet.yaml

  python -m dlio_benchmark.auto_config compare \\
      --original-trace ./traces_original \\
      --dlio-trace ./traces_dlio \\
      --data-roots /data/imagenet
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _cmd_run(args: argparse.Namespace) -> int:
    from .instrument import DFTracerInstrumenter
    from .parser import parse_traces
    from .extractor import SchemaExtractor
    from .generator import generate_yaml

    instrumenter = DFTracerInstrumenter(
        data_roots=args.data_roots,
        output_dir=args.trace_dir,
    )
    pfw_files = instrumenter.run(args.command)
    if not pfw_files:
        logger.error("No trace files produced. Is dftracer installed?")
        return 1

    events = parse_traces(pfw_files, data_roots=args.data_roots)
    schema = SchemaExtractor(data_roots=args.data_roots).extract(events)
    generate_yaml(schema, args.output, workload_name=args.name)
    logger.info(f"Config written to {args.output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .parser import find_trace_files, parse_traces
    from .extractor import SchemaExtractor
    from .generator import generate_yaml

    pfw_files = find_trace_files(args.trace_dir)
    if not pfw_files:
        logger.error(f"No .pfw files found in {args.trace_dir}")
        return 1

    events = parse_traces(pfw_files, data_roots=args.data_roots)
    schema = SchemaExtractor(data_roots=args.data_roots).extract(events)
    generate_yaml(schema, args.output, workload_name=args.name)
    logger.info(f"Config written to {args.output}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error: {e}")
        return 1

    errors = []
    # Check required top-level sections
    for section in ("dataset", "reader", "train"):
        if section not in cfg:
            errors.append(f"Missing required section: {section}")

    # Check dataset sub-fields
    ds = cfg.get("dataset", {})
    for field in ("format", "num_files_train", "record_length"):
        if field not in ds:
            errors.append(f"dataset.{field} is missing")

    if errors:
        for e in errors:
            logger.error(f"  {e}")
        return 1

    logger.info(f"Config {config_path} is valid")
    if "dataset" in cfg:
        logger.info(f"  format:           {cfg['dataset'].get('format')}")
        logger.info(f"  num_files_train:  {cfg['dataset'].get('num_files_train')}")
        logger.info(f"  record_length:    {cfg['dataset'].get('record_length')}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .parser import find_trace_files, parse_traces
    from .compare import compare_traces

    orig_files = find_trace_files(args.original_trace)
    dlio_files  = find_trace_files(args.dlio_trace)

    if not orig_files:
        logger.error(f"No trace files in {args.original_trace}")
        return 1
    if not dlio_files:
        logger.error(f"No trace files in {args.dlio_trace}")
        return 1

    orig_events = parse_traces(orig_files, data_roots=args.data_roots)
    dlio_events  = parse_traces(dlio_files, data_roots=args.data_roots)

    report = compare_traces(orig_events, dlio_events)
    print(report.summary())

    if args.output:
        Path(args.output).write_text(report.summary() + "\n")
        logger.info(f"Report saved to {args.output}")

    return 0 if report.overall_score >= 0.7 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dlio_benchmark.auto_config",
        description="Auto-generate DLIO configs from dftracer workload traces",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- run --
    run = sub.add_parser("run", help="Instrument a workload and generate a DLIO config")
    run.add_argument("--command", required=True, help="Shell command to instrument")
    run.add_argument("--data-roots", nargs="+", required=True, metavar="DIR",
                     help="Dataset root directories to trace")
    run.add_argument("--trace-dir", default="./auto_config_traces",
                     help="Directory to write .pfw trace files (default: ./auto_config_traces)")
    run.add_argument("--output", default="dlio_auto.yaml", help="Output YAML path")
    run.add_argument("--name", default="auto_generated", help="Workload name in config")

    # -- analyze --
    analyze = sub.add_parser("analyze", help="Parse existing traces and generate a DLIO config")
    analyze.add_argument("--trace-dir", required=True, help="Directory containing .pfw files")
    analyze.add_argument("--data-roots", nargs="+", default=None, metavar="DIR",
                         help="Dataset root directories (for filtering)")
    analyze.add_argument("--output", default="dlio_auto.yaml", help="Output YAML path")
    analyze.add_argument("--name", default="auto_generated", help="Workload name in config")

    # -- validate --
    validate = sub.add_parser("validate", help="Validate a DLIO YAML config")
    validate.add_argument("--config", required=True, help="Path to DLIO YAML config")

    # -- compare --
    compare = sub.add_parser("compare", help="Compare original vs DLIO replay trace")
    compare.add_argument("--original-trace", required=True, metavar="DIR",
                         help="Directory with original workload .pfw files")
    compare.add_argument("--dlio-trace", required=True, metavar="DIR",
                         help="Directory with DLIO replay .pfw files")
    compare.add_argument("--data-roots", nargs="+", default=None, metavar="DIR")
    compare.add_argument("--output", default=None, help="Save report to file")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "run":      _cmd_run,
        "analyze":  _cmd_analyze,
        "validate": _cmd_validate,
        "compare":  _cmd_compare,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
