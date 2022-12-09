Tested systems
================
So far we have tested DLIO on the following systems: 
  * Personal workstation, laptops including both MacOSX and Linux OS system. 
  * Supercomputers (Linux), such as Theta @ ALCF, Summit @ OLCF

Limitations and future works
===================================

* DLIO currently assumes the samples to always be 2D images, even though one can set the size of each sample through ```--record_length```. We expect the shape of the sample to have minimal impact to the I/O performance. This yet to be validated case by case. We plan to add option to allow specifying the shape of the sample. 

* We assume the data/label pairs are stored in the same file. Storing data and labels in separate files will be supported in future. We also assume that loading data is the dominant I/O. 

* File format support: currently, we only support tfrecord, hdf5, npz, csv, jpg, jpeg. Other data formats, we simply read the entire file into bytes object without decoding it into meaningful data. 

* Data Loader support: we support reading datasets using TensorFlow tf.data data loader, PyTorch DataLoader, and a set of custom data readers implemented in ```./reader```. For TensorFlow tf.data data loader, PyTorch DataLoader, the specific support are as follows: 
  - We have complete support for tfrecord format in TensorFlow data loader. 
  - For npz, png, jpeg, we currently only support one sample per file case. Multiple samples per file case will be supported in future. We have limited support for hdf5 format for multiple samples per file cases. 

* Profiler support: Darshan is only supported in LINUX system, and might not work well within container. 
