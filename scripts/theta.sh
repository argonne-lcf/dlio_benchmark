#!/usr/bin/sh
#COBALT -n 8 -A datascience -t 1:00:00 -q debug-cache-quad
##COBALT -n 128 -A datascience -t 3:00:00 -q default --jobname=dlio --attrs mcdram=cache:numa=quad
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`

export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
module load datascience/tensorflow-2.2
NNODES=$COBALT_JOBSIZE
RANKS_PER_NODE=1
NRANKS=$((COBALT_JOBSIZE*RANKS_PER_NODE))
NUM_CORES=64
THREADS_PER_CORE=2
NUM_THREADS=$((NUM_CORES*THREADS_PER_CORE))
PROCESS_DISTANCE=$((NUM_THREADS/RANKS_PER_NODE))

DARSHAN_PRELOAD=/soft/perftools/darshan/darshan-3.1.8/lib/libdarshan.so

DATA_DIR=/projects/datascience/dhari/dlio_datasets
rm -rf ${DATA_DIR}/*

OPTS=(-f tfrecord -fa multi -nf 1024  -df ${DATA_DIR})

echo "aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE"
# shellcheck disable=SC2068
echo "Options are: ${OPTS[@]}"
#aprun -n 1 -N 1 -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
#python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
#-gd 1 \
#-p 0 \
#-go 1 \
#-k 1
#
#echo "Data Generation Done."

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0


