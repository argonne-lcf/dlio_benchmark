#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-8}
debug=${2:-n}

# Run the benchmark in PyTorch 
horovodrun -np $num_procs python3 src/dlio_benchmark.py --config-name=unet3d ++workflow.debug $debug ++workflow.generate_data=False
