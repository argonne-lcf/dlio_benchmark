#!/bin/bash
num_procs=${1:-8}

horovodrun -np $num_procs python3 src/dlio_benchmark.py --config-name=bert ++dataset.file-prefix="part"
