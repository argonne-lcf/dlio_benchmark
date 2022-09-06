#!/bin/bash

# Default names of directories for generating data and output
# Aligned with dlio argument_parser.py
mkdir -p data
mkdir -p output

CONTAINER_NAME="train_dlio"

# Remove existing container if a previous run was interrupted
if [ "$(docker ps -a | grep $CONTAINER_NAME)" ]
then
	docker rm $CONTAINER_NAME
fi

docker run -it --rm --name=$CONTAINER_NAME --gpus all -v data:/workspace/dlio/data -v output:/workspace/dlio/output dlio:test /bin/bash run_dlio.sh
