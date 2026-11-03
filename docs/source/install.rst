Installation
=============
The installation of DLIO follows the standard python package installation as follows: 

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    pip install .

The base install includes ``pydftracer[dynamo]``, which depends on PyTorch. To use a CPU-only PyTorch wheel, install it first:

.. code-block:: bash

    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install .

For CUDA support, choose the wheel index compatible with your Python version and driver from the `PyTorch installer <https://pytorch.org/get-started/locally/>`_. For example, with a supported CUDA 12.4 environment:

.. code-block:: bash

    python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    python -m pip install '.[cuda]'

The ``cuda`` extra installs NVIDIA DALI as well. The extra cannot choose a PyTorch wheel index, so install the CUDA PyTorch wheel first. Plain ``pip install .`` and ``pip install -r requirements.txt`` do not explicitly request DALI or select a CUDA wheel index. Because ``pydftracer[dynamo]`` depends on PyTorch, preinstall the CPU wheel as shown above when transitive CUDA packages must be excluded.

To install with AIStore support:

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    pip install .[aistore]

One can also build and install the package as follows

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    python setup.py build
    python setup.py install

One can also install the package directly from github

.. code-block:: bash

    pip install git+https://github.com/argonne-lcf/dlio_benchmark.git@main

    
One can build a docker image run DLIO inside a container.  

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    docker build -t dlio .
    docker run -t dlio dlio_benchmark

A prebuilt docker image is available in docker hub (might not be up-to-date)

.. code-block:: bash 

    docker pull docker.io/zhenghh04/dlio:latest
    docker run -t docker.io/zhenghh04/dlio:latest dlio_benchmark

To run interactively in the docker container. 

.. code-block:: bash

    docker run -t docker.io/zhenghh04/dlio:latest bash
    root@30358dd47935:/workspace/dlio# dlio_benchmark
