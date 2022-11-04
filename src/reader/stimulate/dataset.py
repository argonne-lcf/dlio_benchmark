from abc import ABC, abstractmethod
import tensorflow as tf
import h5py
import math
import numpy as np

class HDF5Generator(ABC):

    def __init__(self, hdf5_file,batch_size):
        self._file = hdf5_file
        self._f = None
        self.batch_size = batch_size
        self.step = 1

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self.get_nevents()
        return self._nevents

    @abstractmethod
    def get_nevents(self):
        pass

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    @abstractmethod
    def get_examples(self, start_idx, stop_idx):
        with tf.profiler.experimental.Trace('Read', step_num=start_idx/self.batch_size, _r=1):
            images = self._f['records'][start_idx: stop_idx]
        return images

def my_import(name):
    components = tf.strings.split(name,sep='.')
    mod = __import__(bytes.decode(components[0].numpy()))
    for comp in components[1:]:
        mod = getattr(mod, bytes.decode(comp.numpy()))
    return mod

class HDF5Dataset(tf.data.Dataset):
    def _generator(generator, file_name, batch_size, start_idx_, num_events=-1,dimention = 1,transfer_size =-1):
        """
        make a generator function that we can query for batches
        """
        klass = my_import(generator)
        reader = klass(file_name,batch_size)
        #reader = generator(file_name,batch_size)
        nevents = reader.openf()
        if num_events == -1:
            num_events = nevents
        if transfer_size == -1:
            num_elements = batch_size
        else:
            num_elements = math.ceil(transfer_size / dimention / dimention)
            if num_elements <= batch_size:
                num_elements = batch_size
            else:
                num_elements = num_elements - (num_elements%batch_size)
        if start_idx_ + num_events > nevents:
            num_events = nevents - start_idx_

        num_yields = math.floor(num_elements / batch_size)
        start_idx, stop_idx,last_event = start_idx_, start_idx_+num_elements,start_idx_+num_events - 1
        step = start_idx_/batch_size
        while True:
            if start_idx > last_event:
                reader.closef()
                return
            if stop_idx > last_event:
                stop_idx = last_event
            images = reader.get_examples(start_idx, stop_idx)
            for i in range(num_yields):
                step += 1
                yield_images = images[i * batch_size:(i + 1) * batch_size]
                if np.shape(yield_images) == (batch_size, dimention, dimention):
                    yield step,yield_images
            start_idx, stop_idx = start_idx + num_elements, stop_idx + num_elements



    def __new__(cls,generator, file_name,output_types,output_shapes, batch_size=1, start_idx=0, num_events=-1,dimension = 1, transfer_size =-1):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=output_types,
            output_shapes=output_shapes,
            args=(generator, file_name, batch_size, start_idx, num_events,dimension,transfer_size,)
        )
        return dataset
