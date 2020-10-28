#!/usr/bin/sh
#COBALT -n 128 -A datascience -t 3:00:00 -q default --jobname=dlio_cosmic --attrs mcdram=cache:numa=quad
##COBALT -n 8 -A datascience -t 1:00:00 -q debug-cache-quad

#source /soft/datascience/tensorflow/tf2.2-craympi.sh
source ./setup_tf2.3.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
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

APP_DATA_DIR=${DATA_DIR}/cosmic_2048

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do
  transfer_size=$((i * 4096))
  n=$COBALT_JOBSIZE
  if [ "${i}" = "1" ]; then
    data_type=hdf5
    OPTS=(-f ${data_type} -fa shared -ct 0 -nf 1 -sf 3072000 -rl 40960 -bs 1 -df ${APP_DATA_DIR} -gd 0 -k 1 -p 1 -l /projects/datascience/dhari/tf_logs/cosmic_ts_${transfer_size}_n_${n})
  else
    data_type=hdf5_opt
    OPTS=(-f ${data_type} -fa shared -ct 0 -nf 1 -sf 3072000 -rl 40960 -bs 1 -df ${APP_DATA_DIR} -ts ${transfer_size} -rp 1 -ps 3072000 -gd 0 -k 1 -p 1 -l /projects/datascience/dhari/tf_logs/cosmic_ts_${transfer_size}_n_${n})
  fi

  echo "aprun -n $((n*RANKS_PER_NODE)) -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
  aprun -n $((n*RANKS_PER_NODE))  -N $RANKS_PER_NODE -j $THREADS_PER_CORE -cc depth -e OMP_NUM_THREADS=$PROCESS_DISTANCE -d $PROCESS_DISTANCE python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
done;

