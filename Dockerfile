FROM ubuntu:20.04

# Add contents of the current directory to /workspace/dlio in the container
ADD . /workspace/dlio

WORKDIR /workspace/dlio

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git vim sysstat mpich gcc-10 g++-10 libc6 libhwloc-dev python3.10 python3-pip

RUN python3 -m pip install --upgrade pip
RUN pip install .[test,dlio-profiler]


