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

DATA_DIR=/projects/datascience/dhari/dlio_datasets
rm -rf ${DATA_DIR}/*

echo "aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE"

#IMAGENET
APP_DATA_DIR=${DATA_DIR}/tf

OPTS=(-f tfrecord -fa multi -nf 1024 -sf 1024 -df ${APP_DATA_DIR} -rl 262144)

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0

#Cosmic Tagger
APP_DATA_DIR=${DATA_DIR}/cosmic

OPTS=(-f hdf5 -fa shared -nf 1 -sf 43000 -df ${APP_DATA_DIR} -rl 40960 -ec 1 -cs 2048 )

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0

#FFN
APP_DATA_DIR=${DATA_DIR}/ffn

OPTS=(-f hdf5 -fa shared -nf 1 -sf 4096 -df ${APP_DATA_DIR} -rl 4096)

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0

#cosmoflow
APP_DATA_DIR=${DATA_DIR}/cosmoflow

OPTS=(-f tfrecord -fa multi -nf 1024 -sf 512 -df ${APP_DATA_DIR} -rl 131072)

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0

#candel
APP_DATA_DIR=${DATA_DIR}/candel

OPTS=(-f csv -fa shared -nf 1 -sf 1120 -df ${APP_DATA_DIR} -rl 262144)

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0

#frnn
APP_DATA_DIR=${DATA_DIR}/frnn

OPTS=(-f npz -fa multi -nf 28000 -sf 1024 -df ${APP_DATA_DIR} -rl 2048)

aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} \
-p 1 \
-k 0



