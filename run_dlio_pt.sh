#!/bin/bash

# Example script to optionally generate data and launch DLIO using the PyTorch framework.

num_procs=${1:-1}
generate_data=${2:-n}
debug=${3:-n}

# Generate npz files that we will open with the dataloader
if [ $generate_data == "y" ] || [ $generate_data == "yes" ]; then
    horovodrun -np $num_procs python3 src/dlio_benchmark.py --framework pytorch --data-folder data/ --output-folder output/ --format npz \
        --generate-data yes --generate-only yes --num-files-train 128 --num-files-eval 16 --num-samples 1 --record-length 134217728
fi

# Run the benchmark in PyTorch 
horovodrun -np $num_procs python3 src/dlio_benchmark.py --framework pytorch --data-folder data/ --output-folder output/ --format data_loader --file-access multi \
    --debug $debug --generate-data no --generate-only no --num-files-train 128 --num-files-eval 16 --do-eval yes --eval-after-epoch 5 \
    --eval-every-epoch 2 --eval-time 5 --num-samples 1 --record-length 134217728 --batch-size 8 --keep-files yes\
    --epochs 10 --computation-time 0.5 --read-threads 0
