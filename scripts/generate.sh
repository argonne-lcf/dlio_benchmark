#!/usr/bin/sh

source /soft/datascience/tensorflow/tf2.2-login.sh
export MPICH_MAX_THREAD_SAFETY=multiple
CURRENT_DIR=`pwd`
DLIO_ROOT=`dirname $CURRENT_DIR`
export PYTHONPATH=$DLIO_ROOT:$PYTHONPATH
DATA_DIR=/projects/datascience/dhari/dlio_datasets


#CANDEL
cm=(none gzip bz2 zip xz)
COMM_OPTS=(-f csv -fa shared -nf 1 -sf 1120 -rl 32768 -bs 1 -cl 0 -gd 1 -go 1 -k 1)


for u in "${cm[@]}"
do
APP_DATA_DIR=${DATA_DIR}/candel_$u
echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co $u -df ${APP_DATA_DIR}"
mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co $u -df ${APP_DATA_DIR}
ls ${DATA_DIR}/candel*
done


#gzip=(0 1 2 3 4 5 6 7 8 9)
#
##Generate FFN Data
#APP_DATA_DIR=${DATA_DIR}/ffn_none
#COMM_OPTS=(-f hdf5 -fa shared -nf 1 -sf 43008 -rl 32768 -bs 1 -ec 1 -cs 4096  -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
#
#
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co none"
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS@]} -co none
#
#for u in "${gzip[@]}"
#do
#APP_DATA_DIR=${DATA_DIR}/ffn_gzip_$u
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co gzip -cl $u"
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co gzip -cl $u
#done
#
#APP_DATA_DIR=${DATA_DIR}/ffn_lzf
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co lzf "
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co lzf
#
##Generate Cosmic Data
#
#APP_DATA_DIR=${DATA_DIR}/cosmic_none
#COMM_OPTS=(-f hdf5 -fa shared -nf 1 -sf 6000 -rl 40960 -bs 1 -ec 1 -cs 4096 -df ${APP_DATA_DIR} -gd 1 -go 1 -k 1)
#
#
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co none"
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${OPTS[@]} -co none
#
#for u in "${gzip[@]}"
#do
#APP_DATA_DIR=${DATA_DIR}/cosmic_gzip_$u
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co gzip -cl $u"
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co gzip -cl $u
#done
#
#APP_DATA_DIR=${DATA_DIR}/cosmic_lzf
#echo "mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co lzf "
#mpirun -n 1 python ${DLIO_ROOT}/src/dlio_benchmark.py ${COMM_OPTS[@]} -co lzf

