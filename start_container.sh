#!/bin/bash

# Default names of directories for generating data and output
# Aligned with dlio argument_parser.py and run_dlio.sh
mkdir -p data
mkdir -p output

docker run -it --rm --name=train_dlio --gpus all -v data:/workspace/dlio/data -v output:/workspace/dlio/output dlio:test /bin/bash
