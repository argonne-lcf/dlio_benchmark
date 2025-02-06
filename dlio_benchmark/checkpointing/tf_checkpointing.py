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

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.utility import Profile
import tensorflow as tf

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from dlio_benchmark.common.enumerations import CheckpointLocationType
from dlio_benchmark.utils.utility import DLIOMPI

def get_tf_datatype(datatype):
    if datatype == "f32":
        return tf.dtypes.float32
    if datatype == "f16":
        return tf.dtypes.float16
    if datatype == "f64":
        return tf.dtypes.float64
    if datatype == "i8":
        return tf.dtypes.int8
    if datatype == "bf16": # bfloat16
        return tf.dtypes.bfloat16

dlp = Profile(MODULE_CHECKPOINT)


class TFCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if TFCheckpointing.__instance is None:
            TFCheckpointing.__instance = TFCheckpointing()
        return TFCheckpointing.__instance
    
    @dlp.log_init
    def __init__(self):
        super().__init__("pb")

    @dlp.log
    def get_tensor(self, size, datatype="i8"):
        return tf.random.uniform((size), maxval=100, dtype=get_tf_datatype(datatype))

    @dlp.log
    def save_state(self, suffix, state):
        name = self.get_name(suffix)
        checkpoint = tf.train.Checkpoint()
        checkpoint.mapped = state
        checkpoint.save(name)

    @dlp.log
    def checkpoint(self, epoch, step_number):
        super().checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()