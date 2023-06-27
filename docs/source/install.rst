Installation
=============
DLIO itself should run directly after installing dependence python packages specified in requirement.txt in the code. 

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    pip install .
    
One can build docker image run DLIO inside a docker container.  

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    docker build -t dlio .
    docker run -t dlio dlio_benchmark

A prebuilt docker image is available in docker hub 

.. code-block:: bash 

    docker pull docker.io/zhenghh04/dlio:latest
    docker run -t docker.io/zhenghh04/dlio:latest dlio_benchmark

To run interactively in the docker container. 

.. code-block:: bash

    docker run -t docker.io/zhenghh04/dlio:latest bash
    root@30358dd47935:/workspace/dlio# dlio_benchmark
