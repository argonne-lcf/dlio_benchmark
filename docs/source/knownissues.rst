Tested systems
================
So far we have tested DLIO on personal workstation, laptops (MAC and Linux), supercomputers. 

Limitations and future works
===================================

* DLIO currently assumes the samples to always be 2D images, even though one can set the size of each sample through ```--record_length```. We expect the shape of the sample to have minimal impact to the I/O itself. This yet to be validated for case by case perspective. We plan to add option to allow specifying the shape of the sample. 

* We assume the data/label pairs are stored in the same file. Storing data and labels in separate files will be supported in future.

* File format support: we only support tfrecord, hdf5, npz, csv, jpg, jpeg formats. Other data formats can be extended. 

* Data Loader support: we support reading datasets using TensorFlow tf.data data loader, PyTorch DataLoader, and a set of custom data readers implemented in ./reader. For TensorFlow tf.data data loader, PyTorch DataLoader  
  - We have complete support for tfrecord format in TensorFlow data loader. 
  - For npz, jpg, jpeg, hdf5, we currently only support one sample per file case. In other words, each sample is stored in an independent file. Multiple samples per file case will be supported in future. 

* Profiler support: Darshan is only supported in LINUX system
