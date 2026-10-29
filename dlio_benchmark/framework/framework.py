"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.data_loader.data_loader_factory import DataLoaderFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import Profile, sleep

dlp = Profile(MODULE_AI_FRAMEWORK)

class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass


class Framework(ABC):
    def __init__(self):
        self.args = ConfigArguments.get_instance()
        self.output_folder = self.args.output_folder


    @abstractmethod
    def init_loader(self, format_type, epoch, data_loader=None):
        self.reader_train = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.TRAIN, epoch=epoch)
        self.reader_valid = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.VALID, epoch=epoch)
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root, self.args.framework)

    @abstractmethod 
    def get_type(self):
        pass
    
    @abstractmethod
    def start_framework_profiler(self):
        pass

    @abstractmethod
    def stop_framework_profiler(self):
        pass

    @abstractmethod
    def trace_object(self, string, step, r):
        pass

    def model(self, epoch, batch, computation_time):
        if self._model is None or batch is None:
            sleep(computation_time)
            return None
        return self._train_batch(batch)

    @staticmethod
    def _split_batch(batch) -> Tuple[Any, Optional[Any]]:
        """Extract inputs and an optional target from supported batch layouts."""
        while isinstance(batch, (tuple, list)) and len(batch) == 1:
            batch = batch[0]
        if isinstance(batch, dict):
            inputs = None
            target = None
            for key in ("data", "image", "input", "inputs"):
                if key in batch:
                    inputs = batch[key]
                    break
            for key in ("label", "labels", "target", "targets"):
                if key in batch:
                    target = batch[key]
                    break
            if inputs is None and len(batch) == 1:
                inputs = next(iter(batch.values()))
            if inputs is None:
                raise ValueError(
                    f"Could not find model input in batch keys {tuple(batch)}"
                )
            return inputs, target
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        return batch, None

    @staticmethod
    def _unwrap_tensor_value(value):
        """Unwrap array and tensor-list containers before backend conversion."""
        if hasattr(value, "as_array"):
            value = value.as_array()
        elif hasattr(value, "as_tensor"):
            value = value.as_tensor()
            if hasattr(value, "as_array"):
                value = value.as_array()
        return value

    @classmethod
    @abstractmethod
    def _as_tensor(cls, value):
        """Convert a value into a native backend tensor."""
        raise NotImplementedError

    def _prepare_batch(self, batch):
        """Validate a labeled batch and apply backend dtype and device rules."""
        inputs, target = self._split_batch(batch)
        if target is None:
            raise ValueError("Native model training requires a target")
        return self._prepare_inputs(inputs), self._prepare_target(target)

    @abstractmethod
    def _prepare_inputs(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def _prepare_target(self, target):
        raise NotImplementedError

    def _train_batch(self, batch):
        inputs, target = self._prepare_batch(batch)
        return self._train_step(inputs, target)

    @abstractmethod
    def _train_step(self, inputs, target):
        """Run the backend's forward pass, loss, and optimizer update."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, batch, epoch_number, step, computation_time):
        pass

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        return self.reader_valid

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def is_nativeio_available(self):
        pass
    # Metadata APIs
    def create_node(self, id, exist_ok=False):
        return False

    def get_node(self, id):
        return None

    def walk_node(self, id, use_pattern=False):
        return None

    def delete_node(self, id):
        return False

    # Data APIs
    def put_data(self, id, data, offset=None, length=None):
        return False

    def get_data(self, id, data, offset=None, length=None):
        return None

    def isfile(self, id):
        return False

