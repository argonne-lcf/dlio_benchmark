#!/bin/bash

# Example script to optionally generate data for the PyTorch framework.
num_procs=${1:-1}

horovodrun -np $num_procs python3 src/dlio_benchmark.py --config-name=unet3d ++workflow.generate_data=True ++workflow.train=False 
