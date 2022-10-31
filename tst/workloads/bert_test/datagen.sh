#!/bin/bash

horovodrun -np 1 python3 src/dlio_benchmark.py --format tfrecord --record-length 2500 --num-samples 31353 \
    --generate-data yes --generate-only yes --num-files-train 50 --file-prefix "part"
