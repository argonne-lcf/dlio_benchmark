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

from src.common.enumerations import FormatType, Shuffle, ReadType, FileAccess, Compression, FrameworkType, DataLoaderType
from dataclasses import dataclass

@dataclass
class ConfigArguments:
    __instance = None

    # command line argument
    # Framework to use
    framework: FrameworkType = FrameworkType.TENSORFLOW
    # Dataset format, such as PNG, JPEG
    format: FormatType = FormatType.TFRECORD
    # Shuffle type
    read_shuffle: Shuffle = Shuffle.OFF
    shuffle_size: int = 1024 * 1024
    memory_shuffle: Shuffle =Shuffle.OFF
    read_type: ReadType = ReadType.ON_DEMAND
    file_access: FileAccess = FileAccess.MULTI
    record_length: int = 64 * 1024
    num_files_train: int = 8 
    num_samples_per_file: int = 1
    batch_size: int = 1 
    epochs: int = 1 
    seed_change_epoch: bool = True
    generate_data: bool = False
    generate_only: bool = False
    data_folder: str = "./data"
    output_folder: str = "./output"
    logdir: str = "./logdir"
    log_file: str  = "dlio.log"
    file_prefix: str = "img"
    keep_files: bool = True
    profiling: bool = False
    seed: int = 123
    do_checkpoint: bool = False
    checkpoint_after_epoch: int = 1 
    epochs_between_checkpoints: int = 0
    steps_between_checkpoints: int =0
    transfer_size: int = None
    read_threads: int = 0
    computation_threads: int = 1
    computation_time: float = 0.
    prefetch: bool = False
    prefetch_size: int = 0 
    enable_chunking: bool = False
    chunk_size: int = 0
    compression: Compression = Compression.NONE
    compression_level: int = 4
    debug: bool = False
    total_training_steps: int = -1
    do_eval: bool = False
    batch_size_eval: int = 1
    num_files_eval: int = 0
    eval_time: float = 0.0
    eval_after_epoch: int = 0
    epochs_between_evals: int = 0 
    model_size: int = 10240
    data_loader: DataLoaderType = DataLoaderType.TENSORFLOW

    def __init__(self):
        """ Virtually private constructor. """
        if ConfigArguments.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ConfigArguments.__instance = self

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigArguments.__instance is None:
            ConfigArguments()
        return ConfigArguments.__instance

def load_config(args, config):
    '''
    Override the args by a system config (typically loaded from a YAML file)
    '''
    if 'framework' in config:
        args.framework = FrameworkType(config['framework'])
    if 'logdir' in config:
        args.logdir = config['logdir']
    if 'output_folder' in config:
        args.output_folder = config['output_folder']
    # dataset related settings
    if 'dataset' in config:
        if 'record_length' in config['dataset']:
            args.record_length = config['dataset']['record_length']
        if 'num_files_train' in config['dataset']:
            args.num_files_train = config['dataset']['num_files_train']
        if 'num_files_val' in config['dataset']:
            args.num_files_val = config['dataset']['num_files_val']
        if 'num_samples_per_file' in config['dataset']:
            args.num_samples_per_file = config['dataset']['num_samples_per_file']
        if 'data_folder' in config['dataset']:
            args.data_folder = config['dataset']['data_folder']
        if 'batch_size' in config['dataset']:
            args.batch_size = config['dataset']['batch_size']
        if 'batch_size_eval' in config['dataset']:
            args.batch_size_eval = config['dataset']['batch_size_eval']
        if 'chunk_size' in config['dataset']:
            args.chunk_size = config['dataset']['chunk_size']
        if 'compression' in config['dataset']:
            args.compression = config['dataset']['compression']
        if 'compression_level' in config['dataset']:
            args.compression_level = config['dataset']['compression_level']
        if 'file_prefix' in config['dataset']:
            args.file_prefix = config['dataset']['file_prefix']
        if 'format' in config['dataset']:
            args.format = FormatType(config['dataset']['format'])
        if 'keep_files' in config['dataset']:
            args.keep_files = config['dataset']['keep_files']

    # data loader
    if 'data_loader' in config:
        if 'data_loader' in config['data_loader']:
            args.data_loader = DataLoaderType(config['data_loader']['data_loader'])
        if 'read_threads' in config['data_loader']:
            args.read_threads = config['data_loader']['read_threads']
        if 'computatation_threads' in config['data_loader']:
            args.computatation_threads = config['data_loader']['computatation_threads']
        if 'prefetch' in config['data_loader']:
            args.prefetch = config['data_loader']['prefetch']
        if 'prefetch_size' in config['data_loader']:
            args.prefetch_size = config['data_loader']['prefetch_size']
        if 'read_shuffle' in config['data_loader']:
            args.read_shuffle = config['data_loader']['read_shuffle']
        if 'shuffle_size' in config['data_loader']:
            args.shuffle_size = config['data_loader']['shuffle_size']
        if 'memory_shuffle' in config['data_loader']:
            args.memory_shuffle = config['data_loader']['memory_shuffle']
        if 'read_type' in config['data_loader']:
            args.read_type = config['data_loader']['read_type']
        if 'file_access' in config['data_loader']:
            args.file_access = config['data_loader']['file_access']
        if 'transfer_size' in config['data_loader']:
            args.transfer_size = config['data_loader']['transfer_size']

    # training relevant setting
    if 'train' in config:
        if 'epochs' in  config['train']:
            args.n_epochs = config['train']['epochs']
        if 'total_training_steps' in config['train']:
            args.total_training_steps = config['train']['total_training_steps']
        if 'seed_change_epoch' in config['train']:
            args.seed_change_epoch = config['train']['seed_change_epoch']
        if 'computation_time' in config['train']:
            args.computation_time = config['train']['computation_time']
        if 'eval_time' in config['train']:
            args.eval_time = config['train']['eval_time']
        if 'eval_after_epoch' in config['train']:
            args.eval_after_epoch = config['train']['eval_after_epoch']
        if 'do_eval' in config['train']:
            args.do_eval = config['train']['do_eval']
        if 'seed' in config['train']:
            args.seed = config['train']['seed']
    if 'checkpoint' in config:
        if 'do_checkpoint' in config['checkpoint']:
            args.do_checkpoint = config['checkpoint']['do_checkpoint']
        if 'checkpoint_after_epoch' in config['checkpoint']:
            args.checkpoint_after_epoch = config['checkpoint']['checkpoint_after_epoch']
        if 'epochs_between_checkpoints' in config['checkpoint']:
            args.epochs_between_checkpoints = config['checkpoint']['epochs_between_checkpoints']
        if 'output_folder' in config['checkpoint']:
            args.output_folder = config['checkpoint']['output_folder']
        if 'model_size' in config['checkpoint']:
            args.model_size = config['checkpoint']['model_size']
    if 'workflow' in config:
        if 'generate_data' in config['workflow']:
            args.generate_data = config['workflow']['generate_data']
        if not ('train' in config['workflow'] and config['workflow']['train']):
            args.generate_only = True
        if 'debug' in config['workflow']:
            args.debug = config['workflow']['debug']
        if 'profiling' in config['workflow']:
            args.profiling = config['workflow']['profiling']

        
