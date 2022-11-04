#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-8}
debug=${2:-n}

# Run the benchmark in PyTorch 
horovodrun -np $num_procs python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --framework pytorch \
    --format data_loader --generate-data no --generate-only no --file-access multi --keep-files yes \
    --epochs 2 --do-eval yes --eval-after-epoch 1 --epochs-between-evals 1 --eval-time 1.572 --computation-time 0.59 \
    --num-files-train 300 --num-files-eval 20 --num-samples 1 --batch-size 4 --debug $debug
