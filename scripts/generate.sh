#!/usr/bin/sh

source /soft/datascience/tensorflow/tf2.2-login.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
DATA_DIR=/projects/datascience/dhari/dlio_datasets

gzip=(0 1 2 3 4 5 6 7 8 9)

#Generate FFN Data
APP_DATA_DIR=${DATA_DIR}/ffn_none
OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs 4096 -co none -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}

for u in "${gzip[@]}"
do
APP_DATA_DIR=${DATA_DIR}/ffn_gzip_$u
OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs 4096 -co gzip -cl $u -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
done

APP_DATA_DIR=${DATA_DIR}/ffn_lzf
OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs 4096 -co lzf -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}

#Generate Cosmic Data

APP_DATA_DIR=${DATA_DIR}/cosmic_none
OPTS=(-f hdf5 -fa shared -nf 1 -sf 6000 -rl 40960 -bs 1 -ec 1 -cs 4096 -co none -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}

for u in "${gzip[@]}"
do
APP_DATA_DIR=${DATA_DIR}/cosmic_gzip_$u
OPTS=(-f hdf5 -fa shared -nf 1 -sf 6000 -rl 40960 -bs 1 -ec 1 -cs 4096 -co gzip -cl $u -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}
done

APP_DATA_DIR=${DATA_DIR}/cosmic_lzf
OPTS=(-f hdf5 -fa shared -nf 1 -sf 6000 -rl 40960 -bs 1 -ec 1 -cs 4096 -co lzf -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]}

