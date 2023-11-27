Profiling 
==========================
We have a built in support for DLIO profiler, https://github.com/hariharan-devarajan/dlio-profiler. A profiler developed for capturing I/O calls. If DLIO profiler is enabled, profiling trace will be generated at the end of the run. The profiler provides profiling information at both application levels and system I/O calls level. 

To enable this functionality, one has to install DLIO profiler throught 
```bash 
pip install dlio-profiler
```
or 
```bash
 git clone git@github.com:hariharan-devarajan/dlio-profiler.git
cd dlio-profiler
python setup.py build
python setup.py install
```
Then set ```DLIO_PROFILER_ENABLE=1``` to enable it. Other environemnt variables setting can be found here: https://dlio-profiler.readthedocs.io/en/latest/api.html#configurations-of-dlio-profiler. 

This will generate tracing files in the output folder, trace-$rank-of-$nproc.pfw, which can then be visualized interactively through https://ui.perfetto.dev/. Below shows one example for UNet3D model.  

.. image:: images/dlio.png

