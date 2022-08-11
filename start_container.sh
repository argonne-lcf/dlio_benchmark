#!/bin/bash

mkdir results
docker run -it --gpus all -v results:/workspace/dlio/results dlio:test /bin/bash