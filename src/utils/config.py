"""
   Copyright (c) 2022, UChicago Argonne, LLC
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

from typing import List, ClassVar
from src.common.enumerations import StorageType, FormatType, Shuffle, ReadType, FileAccess, Compression, FrameworkType, DataLoaderType, Profiler
from dataclasses import dataclass, field
import hydra
import os
import logging

@dataclass
class ConfigArguments:
    __instance = None

    # command line argument
    # Framework to use
    model: str = "default"
    framework: FrameworkType = FrameworkType.TENSORFLOW
    # Dataset format, such as PNG, JPEG
    format: FormatType = FormatType.TFRECORD
    # Shuffle type
    file_shuffle: Shuffle = Shuffle.OFF
    shuffle_size: int = 1024
    sample_shuffle: Shuffle =Shuffle.OFF
    read_type: ReadType = ReadType.ON_DEMAND
    file_access: FileAccess = FileAccess.MULTI
    # Set root as the current directory by default
    storage_root: str = "./"
    storage_type: StorageType = StorageType.LOCAL_FS
    record_length: int = 64 * 1024
    record_length_stdev: int = 0
    num_files_train: int = 8 
    num_samples_per_file: int = 1
    batch_size: int = 1 
    epochs: int = 1 
    seed_change_epoch: bool = True
    generate_data: bool = False
    generate_only: bool = False
    data_folder: str = "./data/"
    output_folder: str = "./output"
    checkpoint_folder: str = "./checkpoints/"
    log_file: str  = "dlio.log"
    file_prefix: str = "img"
    keep_files: bool = True
    do_profiling: bool = True
    profiler: Profiler = Profiler.IOSTAT
    seed: int = 123
    do_checkpoint: bool = False
    checkpoint_after_epoch: int = 1 
    epochs_between_checkpoints: int = 1
    steps_between_checkpoints: int = -1
    transfer_size: int = None
    read_threads: int = 1
    computation_threads: int = 1
    computation_time: float = 0.
    computation_time_stdev: float = 0.
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
    eval_time_stdev: float = 0.0
    eval_after_epoch: int = 1
    epochs_between_evals: int = 1
    model_size: int = 10240
    data_loader: DataLoaderType = DataLoaderType.TENSORFLOW
    num_subfolders_train: int = 0
    num_subfolders_eval: int = 0
    iostat_devices: ClassVar[List[str]] = []

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
    def validate(self):
        """ validate whether the parameters are set correctly"""
        if (self.do_profiling == True) and (self.profiler == Profiler('darshan')):
            if ('LD_PRELOAD' not in os.environ or os.environ["LD_PRELOAD"].find("libdarshan")==-1):
                raise Exception("Please set darshan runtime library in LD_PRELOAD")
        if (self.framework==FrameworkType.TENSORFLOW and self.data_loader == DataLoaderType.PYTORCH) or (self.framework==FrameworkType.PYTORCH and self.data_loader == DataLoaderType.TENSORFLOW):
            raise Exception("Imcompatible between framework and data_loader setup.")
    def reset(self):
        ConfigArguments.__instance = None

def LoadConfig(args, config):
    '''
    Override the args by a system config (typically loaded from a YAML file)
    '''
    if 'framework' in config:
        args.framework = FrameworkType(config['framework'])
    if 'model' in config:
        ''' 
        most of the time, this won't change the benchmark. But in future we might use 
        as a way to do model specific setting. 
        '''
        args.model = config['model']

    if 'storage' in config:
        if 'storage_type' in config['storage']:
            args.storage_type = StorageType(config['storage']['storage_type'])
        if 'storage_root' in config['storage']:
            args.storage_root = config['storage']['storage_root']

    # dataset related settings
    if 'dataset' in config:
        if 'record_length' in config['dataset']:
            args.record_length = config['dataset']['record_length']
        if 'record_length_stdev' in config['dataset']:
            args.record_length_stdev = config['dataset']['record_length_stdev']
        if 'num_files_train' in config['dataset']:
            args.num_files_train = config['dataset']['num_files_train']
        if 'num_files_eval' in config['dataset']:
            args.num_files_eval = config['dataset']['num_files_eval']
        if 'num_samples_per_file' in config['dataset']:
            args.num_samples_per_file = config['dataset']['num_samples_per_file']
        if 'data_folder' in config['dataset']:
            args.data_folder = config['dataset']['data_folder']
            args.data_folder = args.data_folder.rstrip('/')
        if 'num_subfolders_train' in config['dataset']:
            args.num_subfolders_train = config['dataset']['num_subfolders_train']
        if 'num_subfolders_eval' in config['dataset']:
            args.num_subfolders_eval = config['dataset']['num_subfolders_eval']
        if 'enable_chunking' in config['dataset']:
            args.enable_chunking = config['dataset']['enable_chunking']
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

    # data reader
    reader=None
    if 'data_reader' in config:
        reader = config['data_reader']
    elif 'reader' in config:
        reader = config['reader']
    if reader is not None:
        if 'data_loader' in reader:
            args.data_loader = DataLoaderType(reader['data_loader'])
        if 'read_threads' in reader:
            args.read_threads = reader['read_threads']
        if 'computatation_threads' in reader:
            args.computatation_threads = reader['computatation_threads']
        if 'batch_size' in reader:
            args.batch_size = reader['batch_size']
        if 'batch_size_eval' in reader:
            args.batch_size_eval = reader['batch_size_eval']            
        if 'prefetch_size' in reader:
            args.prefetch_size = reader['prefetch_size']
        if 'file_shuffle' in reader:
            args.file_shuffle = reader['file_shuffle']
        if 'shuffle_size' in reader:
            args.shuffle_size = reader['shuffle_size']
        if 'sample_shuffle' in reader:
            args.sample_shuffle = reader['sample_shuffle']
        if 'read_type' in reader:
            args.read_type = reader['read_type']
        if 'transfer_size' in reader:
            args.transfer_size = reader['transfer_size']

    # training relevant setting
    if 'train' in config:
        if 'epochs' in  config['train']:
            args.epochs = config['train']['epochs']
        if 'total_training_steps' in config['train']:
            args.total_training_steps = config['train']['total_training_steps']
        if 'seed_change_epoch' in config['train']:
            args.seed_change_epoch = config['train']['seed_change_epoch']
        if 'computation_time' in config['train']:
            args.computation_time = config['train']['computation_time']
        if 'computation_time_stdev' in config['train']:
            args.computation_time_stdev = config['train']['computation_time_stdev']
        if 'seed' in config['train']:
            args.seed = config['train']['seed']
        
    if 'evaluation' in config:
        if 'eval_time' in config['evaluation']:
            args.eval_time = config['evaluation']['eval_time']
        if 'eval_time_stdev' in config['evaluation']:
            args.eval_time_stdev = config['evaluation']['eval_time_stdev']
        if 'eval_after_epoch' in config['evaluation']:
            args.eval_after_epoch = config['evaluation']['eval_after_epoch']
        if 'epochs_between_evals' in config['evaluation']:
            args.epochs_between_evals = config['evaluation']['epochs_between_evals']

    if 'checkpoint' in config:
        if 'checkpoint_folder' in config['checkpoint']:
            args.checkpoint_folder = config['checkpoint']['checkpoint_folder']
            args.checkpoint_folder = args.checkpoint_folder.rstrip('/')
        if 'checkpoint_after_epoch' in config['checkpoint']:
            args.checkpoint_after_epoch = config['checkpoint']['checkpoint_after_epoch']
        if 'epochs_between_checkpoints' in config['checkpoint']:
            args.epochs_between_checkpoints = config['checkpoint']['epochs_between_checkpoints']
        if 'steps_between_checkpoints' in config['checkpoint']:
            args.steps_between_checkpoints = config['checkpoint']['steps_between_checkpoints']
        if 'model_size' in config['checkpoint']:
            args.model_size = config['checkpoint']['model_size']

    if 'workflow' in config:
        if 'generate_data' in config['workflow']:
            args.generate_data = config['workflow']['generate_data']
        if not (('train' in config['workflow']) and config['workflow']['train']):
            args.generate_only = True
        else:
            args.generate_only = False
        if 'debug' in config['workflow']:
            args.debug = config['workflow']['debug']
        if 'evaluation' in config['workflow']:
            args.do_eval= config['workflow']['evaluation']
        if 'checkpoint' in config['workflow']:
            args.do_checkpoint= config['workflow']['checkpoint']
        if 'profiling' in config['workflow']: 
            args.do_profiling = config['workflow']['profiling']

    if 'profiling' in config: 
        if 'profiler' in config['profiling']:
            args.profiler = Profiler(config['profiling']['profiler'])
        if 'iostat_devices' in config['profiling']:
            args.iostat_devices = config['profiling']['iostat_devices']
            if isinstance(args.iostat_devices, str):
                args.iostat_devices = [args.iostat_devices]
