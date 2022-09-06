#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

# Generate the dataset of 500 .npz files, each containing a single sample of size 327680
horovodrun -np 8 python3 src/dlio_benchmark.py --data-folder data/ -f npz -fa shared -nf 500 -sf 1 -rl 327680 -gd 1 -go 1 | tee output/app.log

# Run the benchmark with 10 epochs and 1s computation time
# python3 src/dlio_benchmark.py --data-folder data/ -f npz -fa multi -nf 500 -sf 1 -rl 327680 -bs 8 -k 1 -e 10 -ct 0.2

# Run the benchmark on 8 GPUs
horovodrun -np 8 python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ -f npz -fa multi -nf 500 -sf 1 -rl 327680 -bs 8 -k 1 -e 10 -ct 1 | tee output/app.log
