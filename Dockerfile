FROM python:3.10.8-slim

# Add contents of the current directory to /workspace/dlio in the container
ADD . /workspace/dlio

WORKDIR /workspace/dlio

RUN apt-get update && \
    apt-get install -y git vim sysstat && \
    apt-get install -y mpich

RUN python -m pip install --upgrade pip
RUN pip install  -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/workspace/dlio"
