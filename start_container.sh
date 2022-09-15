#!/bin/bash

# Default names of directories for generating data and output
# Aligned with dlio argument_parser.py and run_dlio.sh
mkdir -p /dl-bench/lhovon/dlio_benchmark/data
mkdir -p /dl-bench/lhovon/dlio_benchmark/output

docker run -it --rm --name=train_dlio --gpus all -v /dl-bench/lhovon/dlio_benchmark/data:/workspace/dlio/data -v /dl-bench/lhovon/dlio_benchmark/output:/workspace/dlio/output dlio:latest /bin/bash
