"""
   Copyright (c) 2024, UChicago Argonne, LLC
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
import os
import math
from abc import ABC, abstractmethod

from dlio_benchmark.common.enumerations import CheckpointLocationType
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI


class BaseCheckpointing(ABC):

    def __init__(self, ext):
        self.ext = ext
        self.args = ConfigArguments.get_instance()
        checkpoint_storage = StorageFactory().get_storage(self.args.storage_type, self.args.checkpoint_folder,
                                                          self.args.framework)
        checkpoint_storage.create_namespace(exist_ok=True)
        rank_to_checkpoint = self.args.my_rank
        if self.args.checkpoint_type == CheckpointLocationType.RANK_ZERO:
            rank_to_checkpoint = 0
        if rank_to_checkpoint == self.args.my_rank:
            self.model_state = None
            if self.args.model_size > 0:
                self.model_state = {"a": self.get_tensor(self.args.model_size)}
            self.optimization_state = None
            if len(self.args.optimization_groups) > 0:
                self.optimization_state = dict()
                tensor_array_size = 0
                for index, state in enumerate(self.args.optimization_groups):
                    if state > 0:
                        self.optimization_state[str(index)] = {'a': self.get_tensor(state),
                                                               'b': self.get_tensor(state)}
                        tensor_array_size += state
                self.optimization_state["combined"] = self.get_tensor(tensor_array_size)
            self.layer_state = None
            if len(self.args.layer_parameters) > 0:
                self.layer_state = dict()
                for index, state in enumerate(self.args.layer_parameters):
                    if state > 0:
                        self.layer_state[str(index)] = self.get_tensor(state // self.args.tensor_parallelism)

    @abstractmethod
    def get_tensor(self, size):
        return []

    @abstractmethod
    def save_state(self, suffix, state):
        pass

    def get_name(self, suffix):
        return os.path.join(self.args.checkpoint_folder, f"{suffix}.{self.ext}")

    def get_layer_index(self, rank, tensor_parallelism, pipeline_parallelism, total_layers):
        if tensor_parallelism > 1:
            total_layers = total_layers + tensor_parallelism
        
        divisible_layers = total_layers - (total_layers % pipeline_parallelism)
        min_layers_per_pipeline = divisible_layers // pipeline_parallelism
        max_layer_per_pipeline = min_layers_per_pipeline + 1
        pipeline_rank = (rank // tensor_parallelism) % pipeline_parallelism
        left_layers = total_layers - divisible_layers
        num_layers_per_pipeline = max_layer_per_pipeline
        if pipeline_rank >= left_layers:
            num_layers_per_pipeline = min_layers_per_pipeline
        if pipeline_rank < left_layers:
            start_layer = pipeline_rank * max_layer_per_pipeline
            end_layer = start_layer + num_layers_per_pipeline - 1
        else:
            start_layer = left_layers * max_layer_per_pipeline + (pipeline_rank - left_layers) * (min_layers_per_pipeline)
            end_layer = start_layer + num_layers_per_pipeline - 1
        return start_layer, end_layer
    
    @abstractmethod
    def checkpoint(self, epoch, step_number):
        my_rank = DLIOMPI.get_instance().rank()
        rank_to_checkpoint = my_rank
        if self.args.checkpoint_type == CheckpointLocationType.RANK_ZERO:
            rank_to_checkpoint = 0
        if rank_to_checkpoint == my_rank:
            if self.model_state:
                self.save_state(suffix=f"model-{epoch}-{step_number}-{my_rank}", state=self.model_state)
            if self.optimization_state:
                self.save_state(suffix=f"optimizer-{epoch}-{step_number}-{my_rank}", state=self.optimization_state)
            
            start_layer, end_layer = self.get_layer_index(my_rank,self.args.tensor_parallelism, self.args.pipeline_parallelism, self.args.num_layers)
            for layer_index in range(start_layer, end_layer + 1):
                self.save_state(suffix=f"layer-{layer_index}-{epoch}-{step_number}-{my_rank}", state=self.layer_state)

    @abstractmethod
    def finalize(self):
        pass