#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-8}
debug=${2:-n}

# Run the benchmark in PyTorch 
horovodrun -np $num_procs python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --framework pytorch \
    --format data_loader --generate-data no --generate-only no --file-access multi --keep-files yes \
    --epochs 10 --do-eval yes --eval-after-epoch 5 --epochs-between-evals 2 --eval-time 11.572 --computation-time 4.59 \
    --num-files-train 3620 --num-files-eval 42 --num-samples 1 --batch-size 4 --debug $debug
