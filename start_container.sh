#!/bin/bash

mkdir results
docker run -it --name=train_dlio --gpus all -v results:/workspace/dlio/results dlio:test /bin/bash
