#!/bin/bash

# Example script to optionally generate data for the PyTorch framework.
num_procs=${1:-1}

horovodrun -np $num_procs python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --format npz \
    --generate-data yes --generate-only yes --keep-files yes --file-access multi --record-length 1145359 \
    --num-samples 128 --num-files-train 3620 --num-files-eval 42
