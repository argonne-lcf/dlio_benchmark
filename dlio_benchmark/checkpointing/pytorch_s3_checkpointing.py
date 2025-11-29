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
import os
import torch
import ctypes
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from s3torchconnector import S3Checkpoint, S3ClientConfig

def get_torch_datatype(datatype):
    if datatype == "fp32":
        return torch.float32
    elif datatype == "fp16":
        return torch.float16
    elif datatype == "fp64":
        return torch.float64
    elif datatype == "int8":
        return torch.int8
    elif datatype == "uint8":
        return torch.uint8
    elif datatype == "bf16": # bfloat16
        return torch.bfloat16
    else:
        raise Exception(f"Invalid datatype {datatype}")
    
dlp = Profile(MODULE_CHECKPOINT)

class PyTorchS3Checkpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchS3Checkpointing.__instance is None:
            PyTorchS3Checkpointing.__instance = PyTorchS3Checkpointing()
        return PyTorchS3Checkpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        super().__init__("pts3")

        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self.args, "storage_options", {}) or {}

        self.access_key_id = storage_options.get("access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        self.secret_access_key = storage_options.get("secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        self.endpoint = storage_options.get("endpoint_url", os.getenv("AWS_ENDPOINT_URL"))
        self.region = storage_options.get("region", os.getenv("AWS_REGION", "us-east-1"))

        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Build connector config, possibly with env overrides
        self.s3_client_config = S3ClientConfig(
            force_path_style=os.getenv("S3_FORCE_PATH_STYLE", "false").lower() == "true",
            max_attempts=int(os.getenv("S3_MAX_ATTEMPTS", "5")),
        )

        # Initialize the S3Checkpoint instance
        self.s3_checkpoint = S3Checkpoint(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

    @dlp.log
    def get_tensor_core(self, length, datatype="int8", randomize=True):
        torch_dtype=get_torch_datatype(datatype)
        if randomize:
            if torch_dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16]:
                return torch.rand(length, dtype=torch_dtype)
            elif torch_dtype == torch.int8:
                return torch.randint(low=-128,high=128, size=(length,), dtype=torch_dtype)
            elif torch_dtype == torch.uint8:
                return torch.randint(low=0, high=256, size=(length,), dtype=torch_dtype)
            else:
                raise Exception(f"Datatype {torch_dtype} cannot be randomized for random tensor generation.")
        else:
            return torch.ones(length, dtype=torch_dtype)

    def set_madvise_mergeable(self, tensor):
        """
        Apply MADV_MERGEABLE to a PyTorch tensor's memory region with alignment handling.

        1. Validates madvise is initialized and the tensor has valid memory pointers
        2. Calculates page-aligned memory boundaries for the tensor
        3. Applies madvise(MADV_MERGEABLE) to the aligned region
        """
        if not self.madvise_ready:
            return False

        try:
            if not (hasattr(tensor, 'data_ptr') and hasattr(tensor, 'untyped_storage')):
                 return False

            ptr_addr = tensor.data_ptr()
            storage = tensor.untyped_storage()

            if storage is None or ptr_addr == 0:
                 return False

            size_bytes = storage.nbytes()
            if size_bytes <= 0:
                return False

        except Exception:
            return False

        page_size = self.madvise_page_size
        start_addr = ptr_addr
        end_addr = ptr_addr + size_bytes

        aligned_start_addr = (start_addr + page_size - 1) // page_size * page_size
        aligned_end_addr = end_addr // page_size * page_size
        aligned_size = aligned_end_addr - aligned_start_addr

        if aligned_size <= 0:
            return False

        try:
            c_ptr = ctypes.c_void_p(aligned_start_addr)
            c_size = ctypes.c_size_t(aligned_size)
            ret = self.madvise_func(c_ptr, c_size, self.madvise_mergeable)

            if ret == 0:
                return True
            else:
                return False

        except Exception:
            return False

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        # Save checkpoint to S3
        with self.s3_checkpoint.writer(name) as writer:
            torch.save(state, writer)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = dict() # clear up
        # Load checkpoint from S3
        with self.s3_checkpoint.reader(name) as reader:
            state = torch.load(reader)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert(len(state.keys())>0)

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()

