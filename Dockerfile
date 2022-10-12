# We may be able to further simplify this Dockerfile, but this works

# Has tensorflow >= 2.2.0
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

# Add contents of the current directory to /workspace/dlio in the container
ADD . /workspace/dlio

# Remove scripts that are used to launch the container
RUN rm /workspace/dlio/start_dlio.sh

WORKDIR /workspace/dlio

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

# Install horovod with support for both tensorflow and pytorch
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir --upgrade --force-reinstall horovod && ldconfig

ENV PYTHONPATH="${PYTHONPATH}:/workspace/dlio"
