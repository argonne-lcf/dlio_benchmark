#!/usr/bin/sh
source /soft/datascience/tensorflow/tf2.2-craympi.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
vals_0=(0 1KB 2KB 4KB 8KB 16KB 32KB 64KB 128KB 256KB 512KB 1MB 2MB 4MB 8MB 16MB)
for u in "${vals_1[@]}"
do
./ffn.sh $u
while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
do
  sleep 2
  echo "Waiting"
done
done
