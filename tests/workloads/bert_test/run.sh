#!/bin/bash

# Test version that runs very quickly to check we get the expected output

num_procs=${1:-8}
debug=${2:-n}

horovodrun -np 1 python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --framework tensorflow \
    --format tfrecord --num-files-train 50 --num-samples 31353 --record-length 2500 --keep-files yes \
    --computation-time 0.968 --batch-size 48  --computation-threads $num_procs --do-checkpoint yes \
    --model-size 40347133 --steps-between-checkpoints 125 --total-training-steps 300 --transfer-size 262144 --read-threads 1
