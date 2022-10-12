#!/bin/bash

# Example script to optionally generate data and launch DLIO using the Tensorflow framework.

num_procs=${1:-1}
generate_data=${2:-n}
debug=${3:-n}

# Generate tfrecords 
if [ $generate_data == "y" ] || [ $generate_data == "yes" ]; then
    horovodrun -np $num_procs python3 src/dlio_benchmark.py --framework tensorflow --data-folder data/ --output-folder output/ --format tfrecord \
        --generate-data yes --generate-only yes --num-files-train 128 --num-files-eval 16 
fi

# Run the benchmark with Tensorflow
horovodrun -np $num_procs python3 src/dlio_benchmark.py --framework tensorflow --data-folder data/ --output-folder output/ --format tfrecord --file-access multi \
    --debug $debug --generate-data no --generate-only no --num-files-train 128 --num-files-eval 16 --do-eval yes --eval-after-epoch 5 \
    --eval-every-epoch 2 --eval-time 5 --num-samples 1 --record-length 134217728 --batch-size 8 --keep-files yes\
    --epochs 10 --computation-time 0.5
