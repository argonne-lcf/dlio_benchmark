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
import logging
import math
import os
import time
from abc import ABC, abstractmethod

from dlio_benchmark.common.enumerations import CheckpointLocationType
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI, utcnow


def get_datatype_size(datatype):
    if datatype == "int8" or datatype == "uint8":
        return 1
    elif datatype == "fp16" or datatype == "bf16":
        return 2
    elif datatype == "fp32":
        return 4
    elif datatype == "fp64":
        return 8
    else:
        raise Exception("Unsupported datatype {datatype}")

class BaseCheckpointing(ABC):

    def __init__(self, ext):
        #TODO(Huihuo): Add support for checkpointing rng states for transformer type of architecture
        self.ext = ext
        self.args = ConfigArguments.get_instance()
        self.checkpoint_storage = StorageFactory().get_storage(self.args.storage_type, self.args.checkpoint_folder,
                                                          self.args.framework)
        self.MPI = DLIOMPI.get_instance()
        self.comm = self.MPI.comm()
        # define parallelism
        self.model_parallelism = self.args.pipeline_parallelism*self.args.tensor_parallelism
        self.data_parallelism = self.args.comm_size//self.model_parallelism
        self.pipeline_parallism_rank = (self.args.my_rank // self.args.tensor_parallelism) % self.args.pipeline_parallelism
        self.tensor_parallism_rank = self.args.my_rank % self.args.tensor_parallelism
        self.data_parallelism_rank = self.args.my_rank // (self.args.pipeline_parallelism*self.args.tensor_parallelism)
        self.model_parallelism_rank = self.args.my_rank%self.model_parallelism
        self.optimization_groups_predefined = False
        self.layer_parameters_predefined = False
        self.checkpoint_storage.create_namespace(exist_ok=True)
        self.rank_to_checkpoint = self.args.my_rank
        self.num_parameters = self.get_num_parameters()
        self.checkpoint_size = 0.0
        model_checkpoint_size = 0.0
        optimizer_checkpoint_size = 0.0
        if self.args.my_rank == 0:
            logging.info(f"{utcnow()} Total number of parameters in the model: {self.num_parameters}")
        if self.args.zero_stage == 0:
            if self.args.my_rank < self.model_parallelism:
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
            start_layer, end_layer = self.get_layer_index()
            if self.layer_parameters_predefined:
                # This is for old code, where the layer parameters are predefined
                self.layer_state = dict()
                layer_state = dict()
                for index, state in enumerate(self.args.layer_parameters):
                    if state > 0:
                        layer_state[str(index)] = self.get_tensor(state // self.args.tensor_parallelism)
                for layer_index in range(start_layer, end_layer + 1):
                    self.layer_state[str(layer_index)] = layer_state  
            elif self.args.num_layers > 0:
                self.layer_state = dict()
                model_checkpoint_size = 0.0
                for layer_index in range(start_layer, end_layer + 1):
                    self.layer_state[str(layer_index)], size = self.get_layer_state(layer_index)
                    #logging.info(f"{utcnow()} {self.args.my_rank} [{start_layer}-{end_layer}]:::{layer_index}: {size/1024./1024./1024:.4f} GB ")
                    model_checkpoint_size += size
                if self.args.my_rank == 0:
                    logging.debug(f"{utcnow()} Layer states defined! {model_checkpoint_size/1024./1024./1024} GB per rank")

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
                            optimizer_checkpoint_size += state * get_datatype_size(self.args.optimizer_datatype)
                            self.optimization_state[str(index)] = self.get_tensor(state, self.args.optimizer_datatype)
            if self.args.my_rank == 0:
                logging.debug(f"{utcnow()} Optimizer state defined: {optimizer_checkpoint_size / 1024./1024./1024} GB per rank")
            # layer state

            
            self.model_state = None
            if self.args.model_size > 0 and self.args.model_type != "transformer":
                self.model_state = {"a": self.get_tensor(self.args.model_size)}
                if self.args.my_rank == 0:
                    logging.info(f"{utcnow()} Model state defined")

        model_checkpoint_size = self.comm.allreduce(model_checkpoint_size)/1024./1024./1024.
        optimizer_checkpoint_size = self.comm.allreduce(optimizer_checkpoint_size)/1024./1024./1024.
        if self.args.zero_stage < 3:
            model_checkpoint_size /= self.data_parallelism
        self.checkpoint_size = model_checkpoint_size + optimizer_checkpoint_size
        if self.args.my_rank == 0:
            logging.info(f"{utcnow()} Model size: {model_checkpoint_size:.4f} GB")
            logging.info(f"{utcnow()} Optimizer state size: {optimizer_checkpoint_size:.4f} GB")
            logging.info(f"{utcnow()} Total checkpoint size: {self.checkpoint_size:.4f} GB")

    @abstractmethod
    def get_tensor(self, length, datatype="int8"):
        return []

    @abstractmethod
    def save_state(self, suffix, state, fsync=False):
        pass

    @abstractmethod
    def load_state(self, suffix, state):
        pass

    def get_name(self, suffix):
        return os.path.join(self.args.checkpoint_folder, f"{suffix}.{self.ext}")

    def get_num_parameters(self):
        if self.args.hidden_size <= 0:
            return 0
        head_size = self.args.hidden_size//self.args.num_attention_heads
        # column dimension of K & V matrix
        dim_kv = head_size * self.args.num_kv_heads        
        embedding = self.args.vocab_size*self.args.hidden_size
        input_norm = self.args.hidden_size
        # number of elements in Q, K, V attention matrices
        qkv = self.args.hidden_size * (self.args.hidden_size + 2*dim_kv)
        dense = self.args.hidden_size*self.args.hidden_size
        layer_norm = self.args.hidden_size
        # number of parameters from the two MLP layers: h_to_4h and 4h_to_h
        mlp_h_to_4h = self.args.ffn_hidden_size*2*self.args.hidden_size # the factor of 2 is because of gated linear unit                                                                           
        mlp_4h_to_h = self.args.ffn_hidden_size*self.args.hidden_size
        weight = self.args.hidden_size
        # number of parameters from the lm_head layer
        lm_head = embedding
        return embedding  + (input_norm + qkv + dense + layer_norm + mlp_h_to_4h + mlp_4h_to_h)*self.args.num_layers + weight + lm_head

    def get_layer_parameters(self, layer_index):
        head_size = self.args.hidden_size//self.args.num_attention_heads
        # column dimension of K and V matrix
        dim_kv = head_size * self.args.num_kv_heads
        if len(self.args.layer_parameters) > 0:
            self.layer_parameters_predefined = True
            return self.args.layer_parameters
        else:
            if self.args.num_layers <= 0:
                return []
            if self.args.zero_stage < 3:
                sharding_factor = 1
            else:
                sharding_factor = self.data_parallelism
            if layer_index == 0 or layer_index == self.args.num_layers + 1:
                return [self.args.hidden_size * self.args.vocab_size // self.args.tensor_parallelism // sharding_factor] # embedding or lm_head
            elif layer_index == self.args.num_layers + 2:
                return [self.args.hidden_size //sharding_factor]
            else:
                return [ self.args.hidden_size // sharding_factor, # input_norm, 
                        self.args.hidden_size*(self.args.hidden_size+2*dim_kv)//self.args.tensor_parallelism//sharding_factor, # self_attn - this is the 
                        self.args.hidden_size*self.args.hidden_size//self.args.tensor_parallelism//sharding_factor, # dense - this is the o matrix
                        self.args.hidden_size//sharding_factor, # layer_norm
                        self.args.hidden_size*2*self.args.ffn_hidden_size//self.args.tensor_parallelism//sharding_factor, # ffn_h_to_4h, 2 is from gated linear unit
                        self.args.hidden_size*self.args.ffn_hidden_size//self.args.tensor_parallelism//sharding_factor, # ffn_4h_to_h
                ]
    def get_layer_state(self, layer_index):
        layer_parameters = self.get_layer_parameters(layer_index)
        layer_state = dict()
        size = 0.0
        for index, state in enumerate(layer_parameters):
            if state > 0:
                layer_state[str(index)] = self.get_tensor(state, self.args.model_datatype)
                size += state*get_datatype_size(self.args.model_datatype)
        return layer_state, size

    def get_optimization_groups(self):
        if len(self.args.optimization_groups) > 0:
            self.optimization_groups_predefined = True
            return self.args.optimization_groups
        else:
            if self.args.num_layers <= 0:
                return []
            if self.args.zero_stage > 0:
                # zero stage 1, 2, 3
                num_parameters = self.get_num_parameters() // self.args.comm_size
            else:
                # if zero is not used. Only the first data parallel instance will save the optimizer states
                num_parameters= self.get_num_parameters() // self.model_parallelism
            if num_parameters> 0:
                return [num_parameters, self.args.hidden_size*5, 
                        num_parameters, self.args.hidden_size*5, 
                        num_parameters, self.args.hidden_size*5]   
            else:
                return []                                                                                                           

    def get_layer_index(self):
        '''
        The layers indcies are [0, 1, ..., l, l+1, l+2], where l is the total number of transformer layers.                                               
        Layer 0, and layer l+1, l+2 are embedding, lm_head, and weight layers, respectively, they are not part of the transformer layers.                 
        The transformer layers are from 1 to l. We only distribute the transformer layers among the ranks.                                                
        We assume layer 0 is always on rank 0, and l+1 and l+2 are on the last rank.                                                                      
        '''
        pipeline_rank = self.pipeline_parallism_rank
        num_layers_per_pipeline_group = self.args.num_layers//self.args.pipeline_parallelism
        remainder = self.args.num_layers%self.args.pipeline_parallelism
        if pipeline_rank < remainder:
            start_layer = pipeline_rank * (num_layers_per_pipeline_group + 1) + 1
            end_layer = start_layer + num_layers_per_pipeline_group
        else:
            start_layer = remainder * (num_layers_per_pipeline_group + 1) + (pipeline_rank - remainder) * num_layers_per_pipeline_group + 1
            end_layer = start_layer + num_layers_per_pipeline_group - 1
        if not self.layer_parameters_predefined: 
            # will turn this on for all the cases in future
            if pipeline_rank == self.args.pipeline_parallelism - 1:
                end_layer = self.args.num_layers + 2
            if pipeline_rank == 0:
                start_layer = 0
        return start_layer, end_layer
    
    @abstractmethod
    def save_checkpoint(self, epoch, step_number):
        my_rank = DLIOMPI.get_instance().rank()
        start_layer, end_layer = self.get_layer_index()
        # create a specifc folder for each step
        checkpoint_id = f"global_epoch{epoch}_step{step_number}"
        self.checkpoint_storage.create_node(checkpoint_id, exist_ok=True)
        if self.rank_to_checkpoint == my_rank:
            if self.model_state:
                self.save_state(suffix=f"{checkpoint_id}/model_states-{my_rank}", state=self.model_state, fsync = self.args.checkpoint_fsync)

            if self.layer_state:
                start_time = time.time()
                if self.args.zero_stage < 3 and self.args.zero_stage > 0:
                    # if pp is turned on, we assume that the model is sharded across the pipeline stages
                    if self.data_parallelism_rank == 0 and self.args.num_layers > 0:
                        # in this case, model is saved layer by layer
                        if self.args.pipeline_parallelism > 1:
                            for layer_index in range(start_layer, end_layer + 1):
                                self.save_state(suffix=f"{checkpoint_id}/layer_{layer_index}-model_{self.model_parallelism_rank}_model_states", state=self.layer_state[str(layer_index)], fsync = self.args.checkpoint_fsync)
                        else:
                            self.save_state(suffix=f"{checkpoint_id}/model_{self.model_parallelism_rank}_model_states", state=self.layer_state, fsync = self.args.checkpoint_fsync)
                else:
                    # in this case, model is sharded across the data parallel ranks
                    assert(self.args.pipeline_parallelism == 1)
                    self.save_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.data_parallelism_rank}_mp_rank_{self.model_parallelism_rank}_model_states", state=self.layer_state, fsync = self.args.checkpoint_fsync)
                save_model_time = time.time() - start_time
                if my_rank == 0:
                    logging.info(f"{utcnow()} Saved model checkpoint in {save_model_time:.4f} seconds")
                
            if self.optimization_state:
                start_time = time.time()
                self.save_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.data_parallelism_rank}_mp_rank_{self.model_parallelism_rank}_optim_states", state=self.optimization_state, fsync = self.args.checkpoint_fsync)
                save_optimizer_time = time.time() - start_time
                if my_rank == 0:
                    logging.info(f"{utcnow()} Saved optimizer checkpoint in {save_optimizer_time:.4f} seconds")

    @abstractmethod
    def load_checkpoint(self, epoch, step_number):
        my_rank = (DLIOMPI.get_instance().rank() + self.args.checkpoint_load_rank_shift) % DLIOMPI.get_instance().size()
        start_layer, end_layer = self.get_layer_index()
        # create a specifc folder for each step
        checkpoint_id = f"global_epoch{epoch}_step{step_number}"
        self.checkpoint_storage.create_node(checkpoint_id, exist_ok=True)
        if self.rank_to_checkpoint == my_rank:
            
            if self.model_state:
                self.load_state(suffix=f"{checkpoint_id}/model_states-{my_rank}", state=self.model_state, fsync = self.args.checkpoint_fsync)
            
            if self.layer_state:
                start_time = time.time()
                if self.args.zero_stage < 3 and self.args.zero_stage > 0:
                    # if pp is turned on, we assume that the model is sharded across the pipeline stages
                    if self.data_parallelism_rank == 0 and self.args.num_layers > 0:
                        # in this case, model is saved layer by layer
                        if self.args.pipeline_parallelism > 1:
                            for layer_index in range(start_layer, end_layer + 1):
                                self.load_state(suffix=f"{checkpoint_id}/layer_{layer_index}-model_{self.model_parallelism_rank}_model_states", state=self.layer_state[str(layer_index)])
                        else:
                            self.load_state(suffix=f"{checkpoint_id}/model_{self.model_parallelism_rank}_model_states", state=self.layer_state, fsync = self.args.checkpoint_fsync)
                else:
                    # in this case, model is sharded across the data parallel ranks
                    assert(self.args.pipeline_parallelism == 1)
                    self.load_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.data_parallelism_rank}_mp_rank_{self.model_parallelism_rank}_model_states", state=self.layer_state)
                load_model_time = time.time() - start_time
                if my_rank == 0:
                    logging.info(f"{utcnow()} Loaded model checkpoint in {load_model_time:.4f} seconds")
                
            if self.optimization_state:
                start_time = time.time()
                self.load_state(suffix=f"{checkpoint_id}/zero_pp_rank_{self.data_parallelism_rank}_mp_rank_{self.model_parallelism_rank}_optim_states", state=self.optimization_state)   
                load_optimizer_time = time.time() - start_time
                if my_rank == 0:
                    logging.info(f"{utcnow()} Loaded optimizer checkpoint in {load_optimizer_time:.4f} seconds")

    @abstractmethod
    def finalize(self):
        pass