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
from dlio_benchmark.utils.utility import DLIOMPI, utcnow
import logging


class BaseCheckpointing(ABC):

    def __init__(self, ext):
        self.ext = ext
        self.args = ConfigArguments.get_instance()
        self.checkpoint_storage = StorageFactory().get_storage(self.args.storage_type, self.args.checkpoint_folder,
                                                          self.args.framework)
        self.MPI = DLIOMPI.get_instance()
        self.comm = self.MPI.comm()
        # define parallelism
        self.pp = self.args.pipeline_parallelism
        self.tp = self.args.tensor_parallelism
        self.mp = self.tp*self.pp
        self.dp = self.args.comm_size//self.mp
        self.pp_rank = (self.args.my_rank // self.tp) % self.pp
        self.tp_rank = self.args.my_rank % self.tp
        self.dp_rank = self.args.my_rank // (self.pp*self.tp)
        self.mp_rank = self.args.my_rank%self.mp
        self.optimization_groups_predefined = False
        self.layer_parameters_predefined = False
        self.checkpoint_storage.create_namespace(exist_ok=True)
        self.rank_to_checkpoint = self.args.my_rank
        self.num_p = self.get_num_parameters()
        self.checkpoint_size = 0.0
        ss = 0.0
        if self.args.my_rank == 0:
            logging.info(f"{utcnow()} Total number of parameters in the transformation model: {self.num_p}")
        if self.args.zero_stage == -1:
            if self.args.my_rank < self.mp:
                self.rank_to_checkpoint = self.args.my_rank
            else:
                self.rank_to_checkpoint = 0
        if self.rank_to_checkpoint == self.args.my_rank:

            if len(self.args.optimization_groups) > 0:
                self.optimization_groups_predefined = True
            else:
                self.optimization_groups_predefined = False
            if len(self.args.layer_parameters) > 0:
                self.layer_parameters_predefined = True
            else:
                self.layer_parameters_predefined = False


            self.layer_state = None
            start_layer, end_layer = self.get_layer_index(self.args.my_rank, self.tp, self.pp, self.args.num_layers)

            if self.layer_parameters_predefined:
                # This is for old code, where the layer parameters are predefined
                self.layer_state = dict()
                layer_state = dict()
                for index, state in enumerate(self.args.layer_parameters):
                    if state > 0:
                        layer_state[str(index)] = self.get_tensor(state // self.args.tensor_parallelism)
                for layer_index in range(start_layer, end_layer + 1):
                    self.layer_state[str(layer_index)] = layer_state  
            else:
                self.layer_state = dict()
                ss = 0.0
                for layer_index in range(start_layer, end_layer + 1):
                    if self.args.zero_stage < 3:
                        _, size = self.get_layer_state(layer_index)
                    else:
                        self.layer_state[str(layer_index)], size = self.get_layer_state(layer_index)
                    logging.info(f"{utcnow()} {self.args.my_rank}- {layer_index}:
                     {size/1024./1024./1024:.4f} GB ")
                    ss += size
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Layer states defined! {ss/1024./1024./1024} GB per rank")

            # optimization state
            self.optimization_state = None
            optimization_groups = self.get_optimization_groups()
            if len(optimization_groups) > 0:
                self.optimization_state = dict()
                if self.optimization_groups_predefined:
                    # This is for old code, where the optimization groups are predefined, might be deprecated in future
                    tensor_array_size = 0
                    for index, state in enumerate(optimization_groups):
                        if state > 0:
                            self.optimization_state[str(index)] = {'a': self.get_tensor(state),
                                                                'b': self.get_tensor(state)}
                            tensor_array_size += state
                    self.optimization_state["combined"] = self.get_tensor(tensor_array_size)
                else:
                    for index, state in enumerate(optimization_groups):
                        if state > 0:
                            #logging.info(f"{state/1024./1024./1024. } GB")
                            self.checkpoint_size += state
                            self.optimization_state[str(index)] = self.get_tensor(state)
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Optimizer state defined: {self.checkpoint_size / 1024./1024./1024} GB per rank")
            # layer state

            
            self.model_state = None
            if self.args.model_size > 0:
                self.model_state = {"a": self.get_tensor(self.args.model_size)}
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Model state defined")

        ss = self.comm.allreduce(ss)/1024./1024./1024.
        opt = self.comm.allreduce(self.checkpoint_size)/1024./1024./1024.
        if self.args.zero_stage < 3:
            ss /= self.dp
        self.checkpoint_size = ss + opt



        if self.args.my_rank == 0:
            logging.info(f"{utcnow()} Total state size: {ss} GB")
            logging.info(f"{utcnow()} Total checkpoint size: {self.checkpoint_size} GB")
    @abstractmethod
    def get_tensor(self, size):
        return []

    @abstractmethod
    def save_state(self, suffix, state):
        pass

    def get_name(self, suffix):
        return os.path.join(self.args.checkpoint_folder, f"{suffix}.{self.ext}")

    def get_num_parameters(self):
        h, l, ffn, voc = self.args.hidden_size, self.args.num_layers, self.args.ffn_hidden_size, self.args.vocab_size
        embedding = voc*h
        input_norm = h
        qkv = 3*h*h
        dense = h*h
        layer_norm = h
        mlp_h_to_4h = ffn*2*h # the factor of 2 is because of gated linear unit                                                                           
        mlp_4h_to_h = ffn*h
        weight = h
        lm_head = embedding
        return embedding  + (input_norm + qkv + dense + layer_norm + mlp_h_to_4h + mlp_4h_to_h)*l + weight + lm_head

    def get_layer_parameters(self, layer_index):
        dtype_size = 2 # 2 bytes for fp16
        if len(self.args.layer_parameters) > 0:
            self.layer_parameters_predefined = True
            return self.args.layer_parameters
        else:
            if self.args.num_layers <= 0:
                return []
            if self.args.zero_stage < 3:
                sharding_factor = 1
            else:
                sharding_factor = self.dp
            h, l, ffn, voc = self.args.hidden_size, self.args.num_layers, self.args.ffn_hidden_size, self.args.vocab_size
            if layer_index == 0 or layer_index == l + 1:
                return [h * voc // sharding_factor * dtype_size] # embedding or lm_head
            elif layer_index == l + 2:
                return [h//sharding_factor * dtype_size]
            else:
                return [ h // sharding_factor * dtype_size, # input_norm, 
                        h*h*3//self.tp//sharding_factor * dtype_size, # self_attn
                        h*h//self.tp//sharding_factor * dtype_size, # dense
                        h//sharding_factor * dtype_size, # layer_norm
                        h*2*ffn//self.tp//sharding_factor * dtype_size, # ffn_h_to_4h
                        h*ffn//self.tp//sharding_factor * dtype_size, # ffn_4h_to_h
                ]
    def get_layer_state(self, layer_index):
        layer_parameters = self.get_layer_parameters(layer_index)
        layer_state = dict()
        size = 0.0
        for index, state in enumerate(layer_parameters):
            if state > 0:
                layer_state[str(index)] = self.get_tensor(state)
                size += state
        return layer_state, size

    def get_optimization_groups(self):
        h, l, ffn, voc = self.args.hidden_size, self.args.num_layers, self.args.ffn_hidden_size, self.args.vocab_size
        if len(self.args.optimization_groups) > 0:
            self.optimization_groups_predefined = True
            return self.args.optimization_groups
        else:
            if self.args.num_layers <= 0:
                return []
            dtype_size = 4  # 4 bytes for fp32 
            if self.args.zero_stage > 0:
                # zero stage 1, 2, 3
                num_p = self.get_num_parameters() // self.args.comm_size
            else:
                # if zero is not used. Only the first data parallel instance will save the optimizer states
                num_p = self.get_num_parameters() // self.mp
            if num_p > 0:
                return [num_p * dtype_size, 
                        h*5*dtype_size, 
                        num_p * dtype_size, 
                        h*5*dtype_size, 
                        num_p * dtype_size, 
                        h*5*dtype_size, 
                ]   
            else:
                return []                                                                                                           

    def get_layer_index(self, rank, tensor_parallelism, pipeline_parallelism, total_layers):
        '''
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
        '''
        '''
        The layers indcies are [0, 1, ..., l, l+1, l+2], where l is the total number of transformer layers.                                               
        Layer 0, and layer l+1, l+2 are embedding, lm_head, and weight layers, respectively, they are not part of the transformer layers.                 
        The transformer layers are from 1 to l. We only distribute the transformer layers among the ranks.                                                
        We assume layer 0 is always on rank 0, and l+1 and l+2 are on the last rank.                                                                      
        '''
        pipeline_rank = (rank // tensor_parallelism) % pipeline_parallelism
        remainder = total_layers%pipeline_parallelism
        nl = total_layers//pipeline_parallelism
        if pipeline_rank < remainder:
            start_layer = pipeline_rank * (nl + 1) + 1
            end_layer = start_layer + nl + 1
        else:
            start_layer = remainder * (nl + 1) + (pipeline_rank - remainder) * nl + 1
            end_layer = start_layer + nl
        if pipeline_rank == pipeline_parallelism - 1:
            end_layer = total_layers + 2
        if pipeline_rank == 0:
            start_layer = 0
        return start_layer, end_layer
    
    @abstractmethod
    def checkpoint(self, epoch, step_number):
        my_rank = DLIOMPI.get_instance().rank()
        start_layer, end_layer = self.get_layer_index(my_rank,self.args.tensor_parallelism, self.args.pipeline_parallelism, self.args.num_layers)
        # create a specifc folder for each step
        checkpoint_id = f"global_epoch{epoch}_step{step_number}"
        self.checkpoint_storage.create_node(checkpoint_id, exist_ok=True)
        if self.rank_to_checkpoint == my_rank:
            if self.model_state:
                self.save_state(suffix=f"{checkpoint_id}/model_states-{my_rank}", state=self.model_state)

            if self.optimization_state:
                self.save_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.dp_rank}_mp_rank_{self.mp_rank}_optim_states", state=self.optimization_state)                
            
            if self.layer_state:
                if self.args.zero_stage < 3:
                    # if pp is turned on, we assume that the model is sharded across the pipeline stages
                    if self.dp_rank == 0 and self.args.num_layers > 0:
                        # in this case, model is saved layer by layer
                        for layer_index in range(start_layer, end_layer + 1):
                            self.save_state(suffix=f"{checkpoint_id}/layer_{layer_index}-model_{self.mp_rank}_model_states", state=self.get_layer_state(layer_index))
                else:
                    # in this case, model is sharded across the data parallel ranks
                    assert(self.pp == 1)
                    self.save_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.dp_rank}_mp_rank_{self.mp_rank}_model_states", state=self.layer_state)

    @abstractmethod
    def finalize(self):
        pass