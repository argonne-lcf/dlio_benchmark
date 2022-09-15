#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

num_gpus=${1:-1}
generate_data=${2:-n}

# Generate the dataset of 500 .npz files, each containing a single sample of size 327680
# horovodrun -np 8 python3 src/dlio_benchmark.py --data-folder data/ -f tfrecord -fa multi -nf 500 -sf 1 -rl 327680 -gd 1 -go 1
# python3 src/dlio_benchmark.py --data-folder data/ -f tfrecord -fa multi -nf 210 -sf 1 -rl 134217728 -gd 1 -go 1
# sync
# sleep 5

# Run the benchmark with 10 epochs and 1s computation time
# python3 src/dlio_benchmark.py --data-folder data/ -f npz -fa multi -nf 500 -sf 1 -rl 327680 -bs 8 -k 1 -e 10 -ct 0.2

# Run the benchmark - switch generate-data to yes the first time you run 
horovodrun -np $num_gpus python3 src/dlio_benchmark.py --data-folder data/ --output-folder output/ --format tfrecord --file-access multi \
    --generate-data $generate_data --generate-only no --num-files-train 128 --num-files-eval 16 --do-eval yes --eval-after-epoch 5 \
    --eval-every-epoch 2 --eval-time 30 --num-samples 1 --record-length 134217728 --batch-size 8 --keep-files yes\
    --epochs 10 --computation-time 0.5
