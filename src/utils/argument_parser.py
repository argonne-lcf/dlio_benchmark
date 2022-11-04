"""
   Copyright 2021 UChicago Argonne, LLC

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

import argparse

from src.common.enumerations import FormatType, Shuffle, ReadType, FileAccess, Compression, FrameworkType


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ArgumentParser(object):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ArgumentParser.__instance is None:
            ArgumentParser()
        return ArgumentParser.__instance

    def __init__(self):
        super().__init__()
        """ Virtually private constructor. """
        if ArgumentParser.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ArgumentParser.__instance = self
        self.parser = argparse.ArgumentParser(description='DLIO Benchmark')
        self.parser.add_argument("-fr", "--framework", default=FrameworkType.TENSORFLOW, type=FrameworkType,
                                 choices=list(FrameworkType),
                                 help="framework to use.")
        self.parser.add_argument("-f", "--format", default=FormatType.TFRECORD, type=FormatType, choices=list(FormatType),
                                 help="data reader to use.")
        self.parser.add_argument("-r", "--read-shuffle", default=Shuffle.OFF, type=Shuffle, choices=list(Shuffle),
                                 help="Shuffle the list of files to be read.")
        self.parser.add_argument("-ms", "--shuffle-size", default=1024 * 1024, type=int,
                                 help="(TF only) Size of the shuffle buffer in bytes.")
        self.parser.add_argument("-m", "--memory-shuffle", default=Shuffle.OFF, type=Shuffle, choices=list(Shuffle),
                                 help="Shuffle the records returned by the data loader.")
        self.parser.add_argument("-rt", "--read-type", default=ReadType.ON_DEMAND, type=ReadType, choices=list(ReadType),
                                 help="The read behavior for the benchmark.")
        self.parser.add_argument("-fa", "--file-access", default=FileAccess.MULTI, type=FileAccess, choices=list(FileAccess),
                                 help="How the files are accessed in the benchmark.")
        self.parser.add_argument("-rl", "--record-length", default=64 * 1024, type=int,
                                 help="Size of a record/image within dataset")
        self.parser.add_argument("-nf", "--num-files-train", default=8, type=int,
                                 help="Number of files that should be accessed for training.")
        self.parser.add_argument("-sf", "--num-samples", default=1, type=int,
                                 help="Number of samples per file.")
        self.parser.add_argument("-bs", "--batch-size", default=1, type=int,
                                 help="Per worker batch size for training records.")
        self.parser.add_argument("-e", "--epochs", default=1, type=int,
                                 help="Number of epochs to be emulated within benchmark.")
        self.parser.add_argument("-se", "--seed-change-epoch", default=False, type=str2bool,
                                 help="change seed between epochs. y/n")
        self.parser.add_argument("-gd", "--generate-data", default=False, type=str2bool,
                                 help="Enable generation of data. y/n")
        self.parser.add_argument("-go", "--generate-only", default=False, type=str2bool,
                                 help="Only generate files then exit.")
        self.parser.add_argument("-df", "--data-folder", default="./data", type=str,
                                 help="Set the path of folder where data is present in top-level.")
        self.parser.add_argument("-of", "--output-folder", default="./output", type=str,
                                 help="Set the path of folder where output can be generated (checkpoint files and logs)")
        self.parser.add_argument("-lf", "--log-file", default="dlio.log", type=str,
                                 help="Name of the logfile")
        self.parser.add_argument("-fp", "--file-prefix", default="img", type=str,
                                 help="Prefix for generated files.")
        self.parser.add_argument("-k", "--keep-files", default=False, type=str2bool,
                                 help="Keep files after benchmark. y/n")
        self.parser.add_argument("-p", "--profiling",  default=False, type=str2bool,
                                 help="Enable I/O profiling within benchmark. y/n")
        self.parser.add_argument("-l", "--logdir", default="./logdir", type=str,
                                 help="Log Directory for profiling logs.")
        self.parser.add_argument("-s", "--seed", default=123, type=int,
                                 help="The seed to be used shuffling during read/memory.")
        self.parser.add_argument("-c", "--do-checkpoint", default=False, type=str2bool,
                                 help="Enable checkpointing. y/n")
        self.parser.add_argument("-cae", "--checkpoint-after-epoch", default=1, type=int,
                                 help="Epoch number after which to enable checkpointing.")
        self.parser.add_argument("-ebc", "--epochs-between-checkpoints", default=0, type=int,
                                 help="Number of epochs between checkpoints.")
        self.parser.add_argument("-sbc", "--steps-between-checkpoints", default=0, type=int,
                                 help="Number of steps between checkpoints.")
        self.parser.add_argument("-ts", "--transfer-size", default=None, type=int,
                                 help="Transfer Size for tensorflow buffer size.")
        self.parser.add_argument("-tr", "--read-threads", default=0, type=int,
                                 help="Number of threads to be used for reads.")
        self.parser.add_argument("-tc", "--computation-threads", default=1, type=int,
                                 help="Number of threads to be used for pre-processing.")
        self.parser.add_argument("-ct", "--computation-time", default=0, type=float,
                                 help="Processing time (seconds) for each training data batch.")
        self.parser.add_argument("-rp", "--prefetch", default=False, type=str2bool,
                                 help="Enable prefetch within benchmark.")
        self.parser.add_argument("-ps", "--prefetch-size", default=0, type=int,
                                 help="Number of batches to prefetch.")
        self.parser.add_argument("-ec", "--enable-chunking", default=False, type=str2bool,
                                 help="Enable chunking for HDF5 files.")
        self.parser.add_argument("-cs", "--chunk-size", default=0, type=int,
                                 help="Set chunk size in bytes for HDF5.")
        self.parser.add_argument("-co", "--compression", default=Compression.NONE, type=Compression, choices=list(Compression),
                                 help="Compression to use.")
        self.parser.add_argument("-cl", "--compression-level", default=4, type=int,
                                 help="Level of compression for GZip.")
        self.parser.add_argument("-d", "--debug", default=False, type=str2bool,
                                 help="Enable debug in code.")
        self.parser.add_argument("-tts", "--total-training-steps", default=-1, type=int,
                                 help="Total number of training steps to take. Used for single epoch training to stop DLIO before it goes through all the data.")

        # Added to support periodic evaluation on a held-out test set
        self.parser.add_argument("-de", "--do-eval", default=False, type=str2bool,
                                 help="If we should simulate evaluation (single rank only for now). See -et, -eae and -eee to configure.")
        self.parser.add_argument("-bse", "--batch-size-eval", default=1, type=int,
                                 help="Per worker batch size for evaluation records.")
        self.parser.add_argument("-nfe", "--num-files-eval", default=0, type=int,
                                 help="Number of files that should be put aside for evaluation. Defaults to zero.")
        self.parser.add_argument("-et", "--eval-time", default=0, type=float,
                                 help="Processing time (seconds) for each evaluation data batch.")
        self.parser.add_argument("-eae", "--eval-after-epoch", default=0, type=int,
                                 help="Epoch number after which to start evaluating")
        self.parser.add_argument("-ebe", "--epochs-between-evals", default=0, type=int,
                                 help="Evaluation frequency: evaluate every x epochs")
        self.parser.add_argument("-mos", "--model-size", default=10240, type=int,
                                 help="Size of the model (for checkpointing) in bytes")

        self.args = self.parser.parse_args()
        self._validate()

    def _validate(self):
        '''
        TODO: MULTI FILES should have files more than nranks
        TODO: SHARED FILE should have file equal to 1
        '''
        pass

