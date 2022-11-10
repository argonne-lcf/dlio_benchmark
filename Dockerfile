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
    apt-get install -y git vim sysstat

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/workspace/dlio"

# Set the timezone in the container to UTC
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
