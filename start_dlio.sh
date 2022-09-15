#!/bin/bash

# Default names of directories for generating data and output
# Aligned with dlio argument_parser.py
mkdir -p /dl-bench/lhovon/dlio_benchmark/data
mkdir -p /dl-bench/lhovon/dlio_benchmark/output

num_gpus=${1:-1} 				# Default option of 1 GPU
generate_data=${2:-n} 			# Default: don't generate data
container_name=${3:-train_dlio}

# Remove existing container if a previous run was interrupted
if [ "$(docker ps -a | grep $container_name)" ]
then
	docker rm $container_name
fi

docker run -it --rm --name=$container_name --gpus $num_gpus -v /dl-bench/lhovon/dlio_benchmark/data:/workspace/dlio/data -v /dl-bench/lhovon/dlio_benchmark/output:/workspace/dlio/output dlio:latest /bin/bash run_dlio.sh $num_gpus $generate_data
