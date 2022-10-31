#!/bin/bash

usage() {
	echo -e "Usage: $0 [OPTIONS] -- [EXTRA ARGS]"
	echo -e "Convenience script to launch the DLIO benchmark.\n"
	echo -e "The given data-generation, run and post-processing scripts will be launched within the container, flushing the caches after data-generation."
	echo -e "If no data-generation script is given, the data is assumed to have previously been generated in the data directory."
	echo -e "If no run or post-processing scripts are given, an interactive session to the container will be started."
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message."
	echo -e "  -dd, --data-dir\t\tDirectory where the training data is read and generated. './data' by default."
	echo -e "  -od, --output-dir\t\tOutput directory for log and checkpoint files. './output' by default."
	echo -e "  -bd, --device\t\t\tAn I/O device to trace, e.g. sda. Can be passed multiple times."
	echo -e "  -im, --image-name\t\tName of the docker image from which to launch the container. Defaults to 'dlio:latest'."
	echo -e "  -c, --container-name\t\tDocker container name. Defaults to 'dlio'."
	echo -e "  -dgs, --datagen-script\tScript to generate the data for this run. If empty, data will be assumed to exist in data-dir."
	echo -e "  -rs, --run-script\t\tScript used to launch DLIO within the container."
	echo -e "  -pps, --postproc-script\tPost-Porcessing script to generate a report from the DLIO output."
	echo -e "  -it, --interactive\t\tPass withouth a value. Launch an interactive session to the container. Default if no run or post-processing scripts are given."
	echo -e "\nExtra args:"
	echo -e "  Any extra arguments passed after after '--' will be passed as is to the DLIO launch script."
	echo ""
}

main() {
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	
	if [ "${EUID:-$(id -u)}" -ne 0 ]
	then
		echo "Run script as root"
		usage
		exit 1
	fi

	# Defaults
	DATA_DIR=${SCRIPT_DIR}/data
	OUTPUT_DIR=${SCRIPT_DIR}/output
	IMAGE_NAME="dlio:latest"
	CONTAINER_NAME="dlio"
	DATAGEN_SCRIPT=
	RUN_SCRIPT=
	POSTPROC_SCRIPT=
	BLKDEVS=()
	INTERACTIVE_SESSION=0

	while [ $# -gt 0 ]; do
	case "$1" in
		-h | --help ) usage; exit 0 ;;
		-od | --output-dir ) OUTPUT_DIR="$2"; shift 2 ;;
		-dd | --data-dir ) DATA_DIR="$2"; shift 2 ;;
		-bd | --blk-dev ) BLKDEVS+="$2 "; shift 2 ;;
		-im | --image-name ) IMAGE_NAME="$2"; shift 2 ;;
		-c | --container-name ) CONTAINER_NAME="$2"; shift 2 ;;
		-dgs | --datagen-script ) DATAGEN_SCRIPT="$2"; shift 2 ;;
		-rs | --run-script ) RUN_SCRIPT="$2"; shift 2 ;;
		-pps | --postproc-script ) POSTPROC_SCRIPT="$2"; shift 2 ;;
		-it | --interactive ) INTERACTIVE_SESSION=1; shift 1 ;;
		-- ) shift; break ;;
		* ) echo "Invalid option $1"; usage; exit 1 ;;
	esac
	done

	EXTRA_ARGS=$@

	mkdir -p $DATA_DIR
	mkdir -p $OUTPUT_DIR
	chmod -R a+rw $DATA_DIR
	chmod -R a+rw $OUTPUT_DIR

	# Remove existing container from a previous run (docker won't let you use the same name otherwise).
	if [ ! -z $CONTAINER_NAME ]
	then
		if [ $(docker ps -a --format "{{.Names}}" | grep "^${CONTAINER_NAME}$") ]
		then
			echo "Container name already used by an existing container. Attempting to remove it."
			# Check if the name conflict comes from a running container, in which case we have to kill it first.
			if [ $(docker ps --format "{{.Names}}" | grep "^${CONTAINER_NAME}$") ]
			then
				docker kill $CONTAINER_NAME 1>/dev/null
				[[ $? != 0 ]] && exit -1
				docker rm $CONTAINER_NAME 2>/dev/null
			else
				# We can now remove the container
				docker rm $CONTAINER_NAME
				[[ $? != 0 ]] && exit -1
			fi
			echo "Successfully removed container."
		fi
		CONTAINER_NAME_ARG="--name $CONTAINER_NAME"
	fi

	# We will launch the container in the background first, then copy datagen and run scripts from host to container. 
	# This removes the need to rebuild the image every time we change the scripts.
	
	# Launch the container in the background
	docker run --ipc=host -it -d --rm $CONTAINER_NAME_ARG \
			-v $DATA_DIR:/workspace/dlio/data \
			-v $OUTPUT_DIR:/workspace/dlio/output \
			$IMAGE_NAME /bin/bash 1>/dev/null

	# Launch the data generation script within the container
	if [ ! -z $DATAGEN_SCRIPT ]
	then
		echo "Launching data generation script"
		docker cp $DATAGEN_SCRIPT $CONTAINER_NAME:/workspace/dlio
		DATAGEN_SCRIPT=$(basename $DATAGEN_SCRIPT)
		docker exec $CONTAINER_NAME /bin/bash $DATAGEN_SCRIPT
		if [ $? != 0 ]; then echo "Error running data generation!" && exit 1; fi
	fi


	if [ -z $RUN_SCRIPT ] && [ -z $POSTPROC_SCRIPT ]
	then
		INTERACTIVE_SESSION=1
	fi

	if [ $INTERACTIVE_SESSION -eq 1 ]
	then
		docker exec -it $CONTAINER_NAME /bin/bash
		exit 0
	elif [ ! -z $RUN_SCRIPT ]
	then
		echo "Flusing caches"
		# Drop caches on the host
		sync && echo 3 > /proc/sys/vm/drop_caches

		# Launch DLIO within the container
		echo "Launching DLIO"

		# We start iostat within the container to avoid the extra dependency 
		# and timezone problems but it could also be launched on host
		docker exec -d $CONTAINER_NAME sh -c "iostat -mdxtcy -o JSON ${BLKDEVS} 1 > /workspace/dlio/output/iostat.json"
		sleep 1
		IOSTAT_PID=$( pidof iostat )

		# In case we were not able to get iostat's PID
		MAX_RETRIES=20
		while [ -z "$IOSTAT_PID" ]
		do
			if [ $MAX_RETRIES == 0 ]; then
				echo "ERROR: Could not get iostat PID. Exiting."
				exit 1
			fi
			MAX_RETRIES=$(( $MAX_RETRIES-1 ))
			sleep 1
			IOSTAT_PID=$( pidof iostat )
		done

		echo "Launched iostat: $IOSTAT_PID"

		docker cp $RUN_SCRIPT $CONTAINER_NAME:/workspace/dlio

		RUN_SCRIPT=$(basename $RUN_SCRIPT)
		docker exec $CONTAINER_NAME /bin/bash $RUN_SCRIPT $EXTRA_ARGS

		echo "Killing iostat"
		kill -SIGINT $IOSTAT_PID
	fi

		# Launch DLIO within the container
	if [ ! -z $POSTPROC_SCRIPT ]
	then
		echo "Post processing results"
		docker cp $POSTPROC_SCRIPT $CONTAINER_NAME:/workspace/dlio
		POSTPROC_SCRIPT=$(basename $POSTPROC_SCRIPT)
		docker exec $CONTAINER_NAME /bin/bash $POSTPROC_SCRIPT
	fi

	# Container is left running if you want to connect to it and check something
}

main $@
