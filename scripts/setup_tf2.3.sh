unset PYTHONPATH
module load intelpython36                                                
export LD_LIBRARY_PATH=/opt/cray/pe/mpt/7.7.14/gni/mpich-intel-abi/16.0/lib:$LD_LIBRARY_PATH
INSTALL_DIR=/soft/datascience/horovod/v0.20.3/
TF_DIR=/soft/datascience/tensorflow/tf2.3-py36-login
TORCH_DIR=/soft/datascience/pytorch/1.4.0-py36-login/lib/python3.6/site-packages
#MXNET_DIR=/soft/datascience/mxnet/1.5.1-py36                                                  
export PYTHONPATH=${TF_DIR}:${TORCH_DIR}:${MXNET_DIR}:${INSTALL_DIR}/lib/python3.6/site-packages:$PYTHONPATH
