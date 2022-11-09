.. DLIO documentation master file

Deep Learning I/O Benchmark
===============================================================

Deep Learning I/O (DLIO) Benchmark is a benchmark suite aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules.

The main features of DLIO include: 
* Easy-to-use and rich configurations through YAML config files to emulate deep learning application's I/O behavior.
* Able to generate synthetic datasets for different deep learning applications.
* Full transparency over emulation of I/O access with logging at different levels.
* Easy to use data generator to test the performance of different data layouts and its impact on the I/O performance.
* Compatible with modern profiling tools such as Tensorboard and Darshan to extract and analyze I/O behavior.
* Support both sequential training and distributed data parallel training workloads.

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   gettingstarted

.. toctree::
   :maxdepth: 2
   :caption: Tested systems and Known issues

   knownissues
   
.. toctree::
   :maxdepth: 2
   :caption: How to contribute

   contribute

.. toctree::
   :maxdepth: 2
   :caption: Resources

   resources.rst

.. toctree::
   :maxdepth: 2
   :caption: Legal

   copyright
   license
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
