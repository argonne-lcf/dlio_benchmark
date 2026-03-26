"""
   Copyright (c) 2026, UChicago Argonne, LLC
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
import os

try:
    import multistorageclient as msc
    MSC_AVAILABLE = True
except ImportError:
    MSC_AVAILABLE = False
    Path = None
    logging.warning(
        "Multi-Storage Client (MSC) not available. "
        "Install with: pip install multi-storage-client"
    )

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from dlio_benchmark.utils.utility import Profile, dft_ai

dlp = Profile(MODULE_CHECKPOINT)


class PyTorchMscCheckpointing(PyTorchCheckpointing):
    """
    PyTorch checkpointing via NVIDIA Multi-Storage Client (MSC).
    """
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchMscCheckpointing.__instance is None:
            PyTorchMscCheckpointing.__instance = PyTorchMscCheckpointing()
        return PyTorchMscCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "ptmsc")
        self.checkpoint_folder = self.args.storage_root

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync=False):
        name = self.get_name(suffix)
        msc.torch.save(state, os.path.join(self.checkpoint_folder, name))

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = msc.torch.load(os.path.join(self.checkpoint_folder, name))
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert len(state.keys()) > 0

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()
