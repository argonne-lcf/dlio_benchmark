"""
   Copyright (c) 2026, Enakta Labs Ltd
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
import torch
from pydaos.torch import Checkpoint as DaosCheckpoint

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT

dlp = Profile(MODULE_CHECKPOINT)


class PyTorchDaosCheckpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchDaosCheckpointing.__instance is None:
            logging.basicConfig(level=logging.INFO)
            PyTorchDaosCheckpointing.__instance = PyTorchDaosCheckpointing()
        return PyTorchDaosCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "pt")

        prefix = self.args.checkpoint_folder
        pool = self.args.checkpoint_daos_pool
        cont = self.args.checkpoint_daos_cont
        chunk_size = self.args.checkpoint_daos_chunk_size
        chunks_limit = self.args.checkpoint_daos_chunks_limit

        logging.info(f"Checkpointing is set to DAOS pool: {pool}, container: {cont}, prefix: {prefix}, chunk_size: {chunk_size} and chunks_limit: {chunks_limit}")
        self.ckpt = DaosCheckpoint(pool, cont, prefix, transfer_chunk_size=chunk_size, chunks_limit=chunks_limit)

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        with self.ckpt.writer(name) as f:
            torch.save(state, f)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = dict()
        with self.ckpt.reader(name) as f:
            state = torch.load(f)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert(len(state.keys())>0)

    @dft_ai.checkpoint.capture
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()
