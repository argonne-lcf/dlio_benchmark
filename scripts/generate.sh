#!/usr/bin/sh
vals_1=(16 32 64 128 256 512)
vals_2=(1 2 4 8 16)

source /soft/datascience/tensorflow/tf2.2-login.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
DATA_DIR=/projects/datascience/dhari/dlio_datasets

for u in "${vals_1[@]}"
do
APP_DATA_DIR=${DATA_DIR}/ffn_${u}KB
size_val=$((u*1024))
OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs $size_val -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
done

for u in "${vals_2[@]}"
do
APP_DATA_DIR=${DATA_DIR}/ffn_${u}MB
size_val=$((u*1024))
OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs $size_val -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
done
