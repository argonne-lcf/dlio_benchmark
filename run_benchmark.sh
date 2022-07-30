#!/bin/bash

# If missing permissions, add yourself to the dlio group

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

# Generate the dataset of 100 .npz files, each containing a single sample of size 40960
python3 src/dlio_benchmark.py -f npz -fa shared -nf 100 -sf 1 -rl 40960 -bs 2 -df tests -gd 1 -go 1

# Run the benchmark with 100 epochs and 0.2 computation time (unclear what units computation time uses)
python3 src/dlio_benchmark.py -f npz -fa multi -nf 100 -sf 1 -rl 40960 -bs 8 -df tests -k 0 -e 10 -ct 0.2
