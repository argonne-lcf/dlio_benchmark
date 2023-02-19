#!/bin/bash
mpirun -np 8 python3 src/dlio_benchmark.py workload=dlrm
cp -r /workspace/dlio/hydra_log/dlrm/* /workspace/dlio/save_spot