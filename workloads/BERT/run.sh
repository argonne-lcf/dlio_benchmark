#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-8}
debug=${2:-n}
batch_size=$(( 6*$num_procs ))

horovodrun -np 1 python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --framework tensorflow \
    --format tfrecord --num-files-train 500 --num-samples 313532 --record-length 2500 --keep-files yes \
    --computation-time 0.968 --batch-size $batch_size  --computation-threads $num_procs --do-checkpoint yes \
    --model-size 4034713312 --debug $debug --steps-between-checkpoints 1250 --total-training-steps 5000 --transfer-size 262144 --read-threads 1
