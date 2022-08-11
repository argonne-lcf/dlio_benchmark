# FROM tensorflow/tensorflow:2.6.1-gpu
# only has tf 2.0.0, need >= 2.2.0
#FROM nvcr.io/nvidia/tensorflow:20.01-tf2-py3
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

ADD . /workspace/dlio
WORKDIR /workspace/dlio

# Hack to avoid this error https://github.com/NVIDIA/nvidia-docker/issues/1632
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt