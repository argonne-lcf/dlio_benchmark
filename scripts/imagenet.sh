#!/usr/bin/sh
#COBALT -n 512 -A datascience -t 3:00:00 -q default --jobname=dlio 
#COBALT --attrs mcdram=cache:numa=quad

##COBALT -n 8 -A datascience -t 1:00:00 -q debug-cache-quad
source /soft/datascience/tensorflow/tf2.2-craympi.sh
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH

NNODES=$COBALT_JOBSIZE
RANKS_PER_NODE=4
NRANKS=$((COBALT_JOBSIZE*RANKS_PER_NODE))
NUM_CORES=64
THREADS_PER_CORE=2
NUM_THREADS=$((NUM_CORES*THREADS_PER_CORE))
PROCESS_DISTANCE=$((NUM_THREADS/RANKS_PER_NODE))

DARSHAN_PRELOAD=/soft/perftools/darshan/darshan-3.1.8/lib/libdarshan.so

DATA_DIR=/projects/datascience/dhari/dlio_datasets

#IMAGENET
APP_DATA_DIR=${DATA_DIR}/imagenet

OPTS=(-f tfrecord -fa multi -nf 1024 -sf 1024 -rl 262144 -ts 1048576 -tr 8 -tc 8 -df ${APP_DATA_DIR}  -gd 0 -k 1 -p 0)

echo "aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
