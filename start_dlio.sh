#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

DATA_DIR=${SCRIPT_DIR}/data
OUTPUT_DIR=${SCRIPT_DIR}/output

# Default names of directories for generating data and output
# Aligned with dlio argument_parser.py
mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR

num_gpus=${1:-8}
container_name=${2:-train_dlio}
generate_data=${3:-n} 			# Default: don't generate data

# Remove existing container if a previous run was interrupted
if [ "$(docker ps -a | grep $container_name)" ]
then
	docker rm $container_name
fi

# Must use ipc=host to launch the container else pytorch dataloader will crash
# https://github.com/ultralytics/yolov3/issues/283#issuecomment-552776535
docker run -it --rm --name=$container_name --ipc=host --gpus $num_gpus -v $DATA_DIR:/workspace/dlio/data -v $OUTPUT_DIR:/workspace/dlio/output dlio:test /bin/bash run_dlio_pt.sh $num_gpus $generate_data
