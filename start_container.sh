#!/bin/bash

docker run -it --gpus all -v ./results:/workspace/dlio/results dlio:test /bin/bash