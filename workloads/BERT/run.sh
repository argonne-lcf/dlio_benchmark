#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-8}
debug=${2:-n}
batch_size=$(( 6*$num_procs ))

horovodrun -np $num_procs python3 src/dlio_benchmark.py --config-name=bert ++dataset.batch_size=$batch_size ++workflow.debug=$debug ++workflow.generate_data=False
