#!/usr/bin/sh
#COBALT -n 4096 -A datascience -t 3:00:00 -q default --jobname=dlio_cosmic_motivation --attrs mcdram=cache:numa=quad
##COBALT -n 8 -A datascience -t 1:00:00 -q debug-cache-quad
#source /soft/datascience/tensorflow/tf2.2-craympi.sh
#source /soft/datascience/tensorflow/tf2.2-login.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
source ./setup_tf2.3.sh
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
#COBALT_JOBSIZE=1
NNODES=$COBALT_JOBSIZE
RANKS_PER_NODE=4
NRANKS=$((COBALT_JOBSIZE*RANKS_PER_NODE))
NUM_CORES=64
THREADS_PER_CORE=2
NUM_THREADS=$((NUM_CORES*THREADS_PER_CORE))
PROCESS_DISTANCE=$((NUM_THREADS/RANKS_PER_NODE))

#DARSHAN_PRELOAD=/soft/perftools/darshan/darshan-3.1.8/lib/libdarshan.so

DATA_DIR=/projects/datascience/dhari/stimulus_dataset


#Cosmic Tagger

for n in 128 256 512 1024 2048; do

  APP_DATA_DIR=${DATA_DIR}/cosmic_${n}
  OPTS=(-f hdf5 -fa shared -nf 1 -sf $((1500*n)) -rl 40960 -ct 0.001 -bs 1 -df ${APP_DATA_DIR} -gd 0 -k 1 -p 1 -l /projects/datascience/dhari/tf_logs/cosmic_${n})
  echo "aprun -n $((n*RANKS_PER_NODE)) -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
  #mpirun -n $NRANKS \
  aprun -n $((n*RANKS_PER_NODE))  -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
  python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} &
done;
wait


#APP_DATA_DIR=${DATA_DIR}/cosmic
#
#
#
#echo "aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
#
##mpirun -n $NRANKS \
#aprun -n $NRANKS -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE \
#-e DXT_ENABLE_IO_TRACE=1 -e LD_PRELOAD=$DARSHAN_PRELOAD \
#python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
