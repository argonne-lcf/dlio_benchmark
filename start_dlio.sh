#!/bin/bash

mkdir results
docker run -it --gpus all -v results:/workspace/dlio/results dlio:base /bin/bash run_dlio.sh
