Installation
=============
DLIO itself should run directly without installation. However it depends on other python packages which are listed in requirements.txt in the github repo. 

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    pip install -r requirements.txt 
    export PYTHONPATH=$PWD/:$PYTHONPATH
    python ./src/dlio_benchmark.py --help