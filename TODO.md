# ToDoList

This document is a local file describing the features we plan to implement

* Split train and validation dataset [This is done on Nov 7]
    - dataset.data_folder will specify the folder on the top level. 
    - create ./train and ./val under the top level. 
    - num_files_train files will be under ./train and num_files_val will be under ./val folder. 
    - provide two data reader: one for train and one for validation, to allow for different setups. 

* Add support for multiple folders
    - num_subfolders_train, num_subfolders_val
    - split the dataset into different folders

* Add readthedocs documentation. We need rich documentation on the code here. 

