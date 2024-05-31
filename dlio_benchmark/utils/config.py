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
import importlib
import inspect
import hydra

import logging
from time import time


from typing import List, ClassVar

from dlio_benchmark.common.constants import MODULE_CONFIG
from dlio_benchmark.common.enumerations import StorageType, FormatType, Shuffle, ReadType, FileAccess, Compression, \
    FrameworkType, \
    DataLoaderType, Profiler, DatasetType, DataLoaderSampler, CheckpointLocationType, CheckpointMechanismType
from dlio_benchmark.utils.utility import DLIOMPI, get_trace_name, utcnow
from dataclasses import dataclass
import math
import os
import numpy as np

from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile, DLIO_PROFILER_ENABLE

dlp = Profile(MODULE_CONFIG)
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
    sample_shuffle: Shuffle = Shuffle.OFF
    read_type: ReadType = ReadType.ON_DEMAND
    file_access: FileAccess = FileAccess.MULTI
    # Set root as the current directory by default
    storage_root: str = "./"
    storage_type: StorageType = StorageType.LOCAL_FS
    record_length: int = 64 * 1024
    record_length_stdev: int = 0
    record_length_resize: int = 0
    num_files_train: int = 8
    num_samples_per_file: int = 1
    batch_size: int = 1
    epochs: int = 1
    seed_change_epoch: bool = True
    generate_data: bool = False
    generate_only: bool = False
    data_folder: str = "./data/"
    output_folder: str = None
    checkpoint_folder: str = "./checkpoints/"
    log_file: str = "dlio.log"
    file_prefix: str = "img"
    keep_files: bool = True
    do_profiling: bool = False
    profiler: Profiler = Profiler.IOSTAT
    seed: int = 123
    do_checkpoint: bool = False
    checkpoint_after_epoch: int = 1
    epochs_between_checkpoints: int = 1
    steps_between_checkpoints: int = -1
    transfer_size: int = None
    read_threads: int = 1
    dont_use_mmap: bool = False
    computation_threads: int = 1
    computation_time: float = 0.
    computation_time_stdev: float = 0.
    preprocess_time: float = 0.
    preprocess_time_stdev: float = 0.
    prefetch_size: int = 2
    enable_chunking: bool = False
    chunk_size: int = 0
    compression: Compression = Compression.NONE
    compression_level: int = 4
    debug: bool = False
    total_training_steps: int = -1
    do_eval: bool = False
    batch_size_eval: int = 1
    num_files_eval: int = 0
    generation_buffer_size: int = 2 * 1073741824  # 2 GB
    eval_time: float = 0.0
    eval_time_stdev: float = 0.0
    eval_after_epoch: int = 1
    epochs_between_evals: int = 1
    checkpoint_type: CheckpointLocationType = CheckpointLocationType.RANK_ZERO
    checkpoint_mechanism: CheckpointMechanismType = CheckpointMechanismType.NONE
    model_size: int = 10240
    optimization_groups: ClassVar[List[int]] = []
    num_layers: int = 1
    layer_parameters: ClassVar[List[int]] = [17371, 24740228]
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    data_loader: DataLoaderType = DataLoaderType.TENSORFLOW.value
    num_subfolders_train: int = 0
    num_subfolders_eval: int = 0
    iostat_devices: ClassVar[List[str]] = []
    data_loader_classname = None
    checkpoint_mechanism_classname = None
    data_loader_sampler: DataLoaderSampler = None
    reader_classname: str = None
    multiprocessing_context: str = "fork"

    # derived fields
    required_samples: int = 1
    total_samples_eval: int = 1
    total_samples_train: int = 1
    file_list_eval: ClassVar[List[str]] = []
    file_list_train: ClassVar[List[str]] = []
    max_dimension: int = 1
    storage = None
    dimension_stdev: float = 0.0
    dimension: int = 1
    training_steps: int = 0
    eval_steps: int = 0
    samples_per_thread: int = 1
    file_map = None
    global_index_map = None
    data_loader_class = None
    reader_class = None
    checkpoint_mechanism_class = None

    def __init__(self):
        """ Virtually private constructor. """
        if ConfigArguments.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.comm_size = DLIOMPI.get_instance().size()
            self.my_rank = DLIOMPI.get_instance().rank()
            ConfigArguments.__instance = self

    def __setstate__(self, state):
        self.__dict__.update(state)
        DLIOMPI.reset()  # in 'fork' case, clear parent's DLIOMPI
        DLIOMPI.get_instance().set_parent_values(self.my_rank, self.comm_size)
        ConfigArguments.__instance = self

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigArguments.__instance is None:
            ConfigArguments()
        return ConfigArguments.__instance

    def configure_dlio_logging(self, is_child=False):
        # with "multiprocessing_context=fork" the log file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the logging library
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            force=True,
            handlers=[
                logging.FileHandler(self.logfile_path, mode="a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
            # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )

    def configure_dlio_profiler(self, is_child=False, use_pid=False):
        # with "multiprocessing_context=fork" the profiler file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the profiler
        if DLIO_PROFILER_ENABLE:
            dlp_trace = get_trace_name(self.output_folder, use_pid)
            if DLIOMPI.get_instance().rank() == 0:
                logging.info(f"{utcnow()} Profiling DLIO {dlp_trace}")
            return PerfTrace.initialize_log(logfile=dlp_trace,
                                                   data_dir=f"{os.path.abspath(self.data_folder)}:"
                                                            f"{self.data_folder}:./{self.data_folder}:"
                                                            f"{self.checkpoint_folder}:./{self.checkpoint_folder}:"
                                                            f"{os.path.abspath(self.checkpoint_folder)}",
                                                   process_id=self.my_rank)
        return None

    def finalize_dlio_profiler(self, dlp_logger):
        if DLIO_PROFILER_ENABLE and dlp_logger:
            dlp_logger.finalize()

    @dlp.log
    def validate(self):
        """ validate whether the parameters are set correctly"""
        if (self.do_profiling == True) and (self.profiler == Profiler('darshan')):
            if ('LD_PRELOAD' not in os.environ or os.environ["LD_PRELOAD"].find("libdarshan") == -1):
                raise Exception("Please set darshan runtime library in LD_PRELOAD")
        if self.format is FormatType.TFRECORD and (self.data_loader is DataLoaderType.PYTORCH):
            raise Exception(f"{self.framework} support for tfrecord is not implemented for {self.data_loader}.")
        if (self.framework == FrameworkType.TENSORFLOW and self.data_loader == DataLoaderType.PYTORCH) or (
                self.framework == FrameworkType.PYTORCH and self.data_loader == DataLoaderType.TENSORFLOW):
            raise Exception("Imcompatible between framework and data_loader setup.")
        if len(self.file_list_train) != self.num_files_train:
            raise Exception(
                f"Expected {self.num_files_train} training files but {len(self.file_list_train)} found. Ensure data was generated correctly.")
        if len(self.file_list_eval) != self.num_files_eval:
            raise Exception(
                f"Expected {self.num_files_eval} evaluation files but {len(self.file_list_eval)} found. Ensure data was generated correctly.")
        if self.data_loader_classname is not None and self.data_loader_sampler is None:
            raise Exception(
                f"For custom data loaders workload.reader.data_loader_sampler needs to be defined as iter or index.")
        if self.read_threads > 1:
            import psutil
            p = psutil.Process()
            cores_available = len(p.cpu_affinity())
            if cores_available < self.read_threads:
                logging.warning(
                    f"Running DLIO with {self.read_threads} threads for I/O but core available {cores_available} "
                    f"are insufficient and can lead to lower performance.")
        if self.num_layers % self.pipeline_parallelism != 0:
            raise Exception(
                f"Expected checkpoint.num_layers {self.num_layers} should be multiple of "
                f"checkpoint.pipeline_parallelism {self.pipeline_parallelism}.")
        if self.num_layers % self.tensor_parallelism != 0:
            raise Exception(
                f"Expected checkpoint.num_layers {self.num_layers} should be multiple of "
                f"checkpoint.tensor_parallelism {self.tensor_parallelism}.")

    @staticmethod
    def reset():
        ConfigArguments.__instance = None

    @dlp.log
    def derive_configurations(self, file_list_train=None, file_list_eval=None):
        self.dimension = int(math.sqrt(self.record_length))
        self.dimension_stdev = self.record_length_stdev / 2.0 / math.sqrt(self.record_length)
        self.max_dimension = self.dimension
        if self.checkpoint_mechanism == CheckpointMechanismType.NONE:
            if self.framework == FrameworkType.TENSORFLOW:
                self.checkpoint_mechanism = CheckpointMechanismType.TF_SAVE
            elif self.framework == FrameworkType.PYTORCH:
                self.checkpoint_mechanism = CheckpointMechanismType.PT_SAVE
        if (self.record_length_resize > 0):
            self.max_dimension = int(math.sqrt(self.record_length_resize))
        if (file_list_train != None and file_list_eval != None):
            self.resized_image = np.random.randint(255, size=(self.max_dimension, self.max_dimension), dtype=np.uint8)
            self.file_list_train = file_list_train
            self.file_list_eval = file_list_eval
            self.num_files_eval = len(file_list_eval)
            self.num_files_train = len(file_list_train)
            self.total_samples_train = self.num_samples_per_file * len(self.file_list_train)
            self.total_samples_eval = self.num_samples_per_file * len(self.file_list_eval)
            self.required_samples = self.comm_size * self.batch_size
            if self.read_threads > 0:
                self.required_samples *= self.read_threads
            self.training_steps = int(math.ceil(self.total_samples_train / self.batch_size / self.comm_size))
            self.eval_steps = int(math.ceil(self.total_samples_eval / self.batch_size_eval / self.comm_size))
        if self.data_loader_sampler is None and self.data_loader_classname is None:
            if self.data_loader == DataLoaderType.TENSORFLOW:
                self.data_loader_sampler = DataLoaderSampler.ITERATIVE
            elif self.data_loader in [DataLoaderType.PYTORCH, DataLoaderType.DALI]:
                self.data_loader_sampler = DataLoaderSampler.INDEX
        if self.data_loader_classname is not None:
            from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
            classname = self.data_loader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.data_loader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseDataLoader):
                    if DLIOMPI.get_instance().rank() == 0:
                        logging.info(f"Discovered custom data loader {class_name}")
                    self.data_loader_class = obj
                    break
        if self.checkpoint_mechanism_classname is not None:
            from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
            classname = self.checkpoint_mechanism_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.checkpoint_mechanism_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseCheckpointing):
                    if DLIOMPI.get_instance().rank() == 0:
                        logging.info(f"Discovered custom checkpointing mechanism {class_name}")
                    self.checkpoint_mechanism_class = obj
                    break
        if self.reader_classname is not None:
            from dlio_benchmark.reader.reader_handler import FormatReader
            classname = self.reader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.reader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, FormatReader):
                    if DLIOMPI.get_instance().rank() == 0:
                        logging.info(f"Discovered custom data reader {class_name}")
                    self.reader_class = obj
                    break

    @dlp.log
    def build_sample_map_iter(self, file_list, total_samples, epoch_number):
        logging.debug(f"ranks {self.comm_size} threads {self.read_threads} tensors")
        num_files = len(file_list)
        num_threads = 1
        if self.read_threads > 0 and self.data_loader is not DataLoaderType.DALI:
            num_threads = self.read_threads
        samples_per_proc = total_samples//self.comm_size            
        self.samples_per_thread = samples_per_proc // num_threads
        file_index = 0
        sample_index = 0
        sample_global_list = np.arange(total_samples)
        if self.file_shuffle is not Shuffle.OFF:
            if self.seed_change_epoch:
                np.random.seed(self.seed + epoch_number)
            else:
                np.random.seed(self.seed)
            np.random.shuffle(sample_global_list)
        process_thread_file_map = {}
        abs_path = os.path.abspath(file_list[file_index])

        for rank in range(self.comm_size):
            if rank not in process_thread_file_map:
                process_thread_file_map[rank] = {}            
            for thread_index in range(num_threads):
                if (thread_index < samples_per_proc%num_threads):
                    addition = 1
                else:
                    addition = 0
                if thread_index not in process_thread_file_map[rank]:
                    process_thread_file_map[rank][thread_index] = []
                selected_samples = 0
                while selected_samples < self.samples_per_thread+addition:
                    process_thread_file_map[rank][thread_index].append((sample_global_list[sample_index],
                                                                        abs_path,
                                                                        sample_global_list[
                                                                        sample_index] % self.num_samples_per_file))

                    sample_index += 1
                    selected_samples += 1
                    if sample_index >= self.num_samples_per_file:
                        sample_index = 0
                        file_index += 1
                        if file_index >= num_files:
                            break
                        abs_path = os.path.abspath(file_list[file_index])
        return process_thread_file_map

    @dlp.log
    def get_global_map_index(self, file_list, total_samples):
        process_thread_file_map = {}
        for global_sample_index in range(total_samples):
            file_index = global_sample_index//self.num_samples_per_file
            abs_path = os.path.abspath(file_list[file_index]) 
            sample_index = global_sample_index % self.num_samples_per_file
            process_thread_file_map[global_sample_index] = (abs_path, sample_index)
        return process_thread_file_map

    @dlp.log
    def reconfigure(self, epoch_number, dataset_type):
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            if self.file_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(self.file_list_train) if dataset_type is DatasetType.TRAIN else np.random.shuffle(
                    self.file_list_eval)
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            if dataset_type is DatasetType.TRAIN:
                global_file_map = self.build_sample_map_iter(self.file_list_train, self.total_samples_train,
                                                             epoch_number)
            else:
                global_file_map = self.build_sample_map_iter(self.file_list_eval, self.total_samples_eval, epoch_number)
            self.file_map = global_file_map[self.my_rank]
        elif self.data_loader_sampler == DataLoaderSampler.INDEX:
            if dataset_type is DatasetType.TRAIN:
                self.global_index_map = self.get_global_map_index(self.file_list_train, self.total_samples_train)
            else:
                self.global_index_map = self.get_global_map_index(self.file_list_eval, self.total_samples_eval)


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
        if 'record_length_resize' in config['dataset']:
            args.record_length_resize = config['dataset']['record_length_resize']
        if 'num_files_train' in config['dataset']:
            args.num_files_train = config['dataset']['num_files_train']
        if 'num_files_eval' in config['dataset']:
            args.num_files_eval = config['dataset']['num_files_eval']
        if 'generation_buffer_size' in config['dataset']:
            args.generation_buffer_size = config['dataset']['generation_buffer_size']
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
    reader = None
    if 'data_reader' in config:
        reader = config['data_reader']
    elif 'reader' in config:
        reader = config['reader']
    if reader is not None:
        if 'dont_use_mmap' in reader:
            args.dont_use_mmap = reader['dont_use_mmap']
        if 'reader_classname' in reader:
            args.reader_classname = reader['reader_classname']
        if 'multiprocessing_context' in reader:
            args.multiprocessing_context = reader['multiprocessing_context']
        if 'data_loader' in reader:
            args.data_loader = DataLoaderType(reader['data_loader'])
        if 'data_loader_classname' in reader:
            args.data_loader_classname = reader['data_loader_classname']
        if 'data_loader_sampler' in reader:
            args.data_loader_sampler = DataLoaderSampler(reader['data_loader_sampler'])
        if 'read_threads' in reader:
            args.read_threads = reader['read_threads']
        if 'computation_threads' in reader:
            args.computation_threads = reader['computation_threads']
        if 'batch_size' in reader:
            args.batch_size = reader['batch_size']
        if 'batch_size_eval' in reader:
            args.batch_size_eval = reader['batch_size_eval']
        if 'prefetch_size' in reader:
            args.prefetch_size = reader['prefetch_size']
        if 'file_shuffle' in reader:
            args.file_shuffle = reader['file_shuffle']
        if 'file_access' in reader:
            args.file_access = FileAccess(reader['file_access'])
        if 'shuffle_size' in reader:
            args.shuffle_size = reader['shuffle_size']
        if 'sample_shuffle' in reader:
            args.sample_shuffle = Shuffle(reader['sample_shuffle'])
        if 'read_type' in reader:
            args.read_type = reader['read_type']
        if 'transfer_size' in reader:
            args.transfer_size = reader['transfer_size']
        if 'preprocess_time' in reader:
            args.preprocess_time = reader['preprocess_time']
        if 'preprocess_time_stdev' in reader:
            args.preprocess_time_stdev = reader['preprocess_time_stdev']

    # training relevant setting
    if 'train' in config:
        if 'epochs' in config['train']:
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
        if 'type' in config['checkpoint']:
            args.checkpoint_type = CheckpointLocationType(config['checkpoint']['type'])
        if 'checkpoint_mechanism_classname' in config['checkpoint']:
            args.checkpoint_mechanism_classname = config['checkpoint']['checkpoint_mechanism_classname']
        if 'model_size' in config['checkpoint']:
            args.model_size = config['checkpoint']['model_size']
        if 'optimization_groups' in config['checkpoint']:
            args.optimization_groups = config['checkpoint']['optimization_groups']
        if 'num_layers' in config['checkpoint']:
            args.num_layers = config['checkpoint']['num_layers']
        if 'layer_parameters' in config['checkpoint']:
            args.layer_parameters = config['checkpoint']['layer_parameters']
        if 'tensor_parallelism' in config['checkpoint']:
            args.tensor_parallelism = config['checkpoint']['tensor_parallelism']
        if 'pipeline_parallelism' in config['checkpoint']:
            args.pipeline_parallelism = config['checkpoint']['pipeline_parallelism']
    if 'output' in config:
        if 'folder' in config['output']:
            args.output_folder = config['output']['folder']
        if 'log_file' in config['output']:
            args.log_file = config['output']['log_file']

    if args.output_folder is None:
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.output_folder = hydra_cfg['runtime']['output_dir']
        except:
            args.output_folder = 'output/'
    args.logfile_path = os.path.join(args.output_folder, args.log_file)

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
            args.do_eval = config['workflow']['evaluation']
        if 'checkpoint' in config['workflow']:
            args.do_checkpoint = config['workflow']['checkpoint']
        if 'profiling' in config['workflow']:
            args.do_profiling = config['workflow']['profiling']

    if 'profiling' in config:
        if 'profiler' in config['profiling']:
            args.profiler = Profiler(config['profiling']['profiler'])
        if 'iostat_devices' in config['profiling']:
            args.iostat_devices = config['profiling']['iostat_devices']
            if isinstance(args.iostat_devices, str):
                args.iostat_devices = [args.iostat_devices]
