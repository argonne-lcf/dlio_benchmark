#!/bin/bash

usage() {
	echo -e "Usage: $0 [OPTIONS] -- [EXTRA ARGS]"
	echo -e "Launches the DLIO container and runs the given script or an interactive session if none is given."
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message."
	echo -e "  -d, --data-dir\t\tDirectory where the training data is read and generated."
	echo -e "  -o, --data-dir\t\tOutput directory for log and checkpoint files."
	echo -e "  -i, --image-name\t\tName of the docker image to launch the container from. Defaults to 'dlio:latest'."
	echo -e "  -c, --container-name\t\tName to give the docker container. Defaults to none."
	echo -e "  -s, ---script\t\t\tScript to launch within the container. Defaults to launching bash within the container."
	echo -e "\nExtra args:"
	echo -e "  Any extra arguments passed after after '--' will be passed as is to the DLIO launch script."
	echo ""
	exit 1
}

main() {

	if [ "${EUID:-$(id -u)}" -ne 0 ]
	then
		echo "Run script as root"
		usage
		exit 1
	fi

	SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

	# Defaults
	DATA_DIR=${SCRIPT_DIR}/data
	OUTPUT_DIR=${SCRIPT_DIR}/output
	IMAGE_NAME="dlio:latest"
	CONTAINER_NAME=
	SCRIPT=

	while [ $# -gt 0 ]; do
	case "$1" in
		-h | --help ) usage ;;
		-o | --output-dir ) OUTPUT_DIR="$2"; shift 2 ;;
		-d | --data-dir ) DATA_DIR="$2"; shift 2 ;;
		-i | --image-name ) IMAGE_NAME="$2"; shift 2 ;;
		-c | --container-name ) CONTAINER_NAME="$2"; shift 2 ;;
		-s | --script ) SCRIPT="$2"; shift 2 ;;
		-- ) shift; break ;;
		* ) echo "Invalid option"; usage ;;
	esac
	done

	EXTRA_ARGS=$@

	mkdir -p $DATA_DIR
	mkdir -p $OUTPUT_DIR

	# Remove existing and inactive container from a previous run (docker won't let you use the same name otherwise).
	# This will fail if the container name is used by a running container so it won't kill someone's running container.
	if [ ! -z $CONTAINER_NAME ]
	then
		if [ "$(docker ps -a | grep $CONTAINER_NAME)" ]
		then
			echo "Container name already in use. Attempting to remove it."
			docker rm $CONTAINER_NAME
			[[ $? != 0 ]] && exit
			echo "Successfully remove container"
		fi
		CONTAINER_NAME_ARG="--name $CONTAINER_NAME"
	fi

	# Must use ipc=host to launch the container else pytorch dataloader will crash
	# https://github.com/ultralytics/yolov3/issues/283#issuecomment-552776535
	docker run -it --rm --ipc=host $CONTAINER_NAME_ARG \
		-v $DATA_DIR:/workspace/dlio/data \
		-v $OUTPUT_DIR:/workspace/dlio/output $IMAGE_NAME /bin/bash $SCRIPT $EXTRA_ARGS
}

main $@