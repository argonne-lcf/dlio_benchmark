# Has tensorflow >= 2.2.0
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

# Add contents of the current directory to /workspace/dlio in the container
ADD . /workspace/dlio
# Remove scripts that should only be used outside of the container (to launch it)
RUN rm /workspace/dlio/start_container.sh
RUN rm /workspace/dlio/start_dlio.sh

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
