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
from numpy import append
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import utcnow, DLIOMPI

import os
import json
import math
import logging
import pandas as pd
from time import time
import numpy as np
import psutil
import platform
import socket
from mpi4py import MPI
def lines_to_dict(lines):
    dict = {}
    for l in lines.split("\n"):
        if len(l.split(":"))==2: 
            k, v = l.split(":")
        dict[k] = v
    return dict

class StatsCounter(object):

    def __init__(self):
        self.comm = DLIOMPI.get_instance().comm()
        self.args = ConfigArguments.get_instance()
        self.my_rank = self.args.my_rank
        self.comm_size = self.args.comm_size
        self.output_folder = self.args.output_folder
        self.record_size = self.args.record_length
        self.batch_size = self.args.batch_size
        self.batch_size_eval = self.args.batch_size_eval
        self.summary = {}
        self.summary['start'] = utcnow()
        self.summary['num_accelerators'] = self.comm_size
        self.summary['num_hosts'] = self.comm_size //DLIOMPI.get_instance().npernode()
        self.summary['hostname'] = socket.gethostname()
        self.summary['metric'] = {}
        self.summary['num_files_train'] = self.args.num_files_train
        self.summary['num_files_eval'] = self.args.num_files_eval
        self.summary['num_samples_per_file'] = self.args.num_samples_per_file
        self.summary['host_cpu_count'] = psutil.cpu_count()
        self.summary['host_processor_name'] = platform.processor()
        self.summary['potential_caching'] = False

        if os.path.exists("/proc/cpuinfo"):
            self.summary['host_cpuinfo'] = lines_to_dict(open("/proc/cpuinfo", "r").read())
        if os.path.exists("/proc/meminfo"):
            self.summary['host_meminfo'] = lines_to_dict(open("/proc/meminfo", "r").read())
        max_steps = math.floor(self.args.num_samples_per_file * self.args.num_files_train / self.args.batch_size / self.args.comm_size)

        if self.args.total_training_steps > 0:
            if self.args.total_training_steps > max_steps:
                logging.error(f"Only have enough data for {max_steps} steps but {self.args.total_training_steps} wanted")
                exit(-1)
            self.steps_override = True
            self.steps = self.args.total_training_steps
        else:
            self.steps_override = False
            self.steps = max_steps
        
        self.steps_eval = math.floor(self.args.num_samples_per_file * self.args.num_files_eval / self.args.batch_size_eval / self.args.comm_size)
        # Only the root process keeps track of overall stats
        if self.my_rank == 0:
            self.per_epoch_stats = {}
        # Each process keeps track of its loading and processing times independently
        self.output = {}
        self.output['host_memory_GB'] = psutil.virtual_memory().total/1024./1024./1024
        DLIOMPI.get_instance().npernode()
        host_memory = np.zeros(DLIOMPI.get_instance().size()//DLIOMPI.get_instance().npernode())
        host_memory_agg = np.zeros(DLIOMPI.get_instance().size()//DLIOMPI.get_instance().npernode())
        if DLIOMPI.get_instance().rank()%DLIOMPI.get_instance().npernode()==0:
            host_memory[DLIOMPI.get_instance().rank()//DLIOMPI.get_instance().npernode()] = self.output['host_memory_GB']
        DLIOMPI.get_instance().comm().Reduce(host_memory, host_memory_agg, op=MPI.SUM, root=0)
        self.summary['host_memory_GB'] = list(host_memory_agg)
        self.output['host_cpu_count'] = psutil.cpu_count()
        self.output['host_processor_name'] = platform.processor()
        self.output['potential_caching'] = False
        if os.path.exists("/proc/cpuinfo"):
            self.output['host_cpuinfo'] = lines_to_dict(open("/proc/cpuinfo", "r").read())
        if os.path.exists("/proc/meminfo"):
            self.output['host_meminfo'] = lines_to_dict(open("/proc/meminfo", "r").read())

        self.train_au = []
        self.eval_au = []
        self.train_throughput = []
        self.eval_throughput = []
        data_per_node = DLIOMPI.get_instance().npernode()*self.args.num_samples_per_file * self.args.num_files_train//DLIOMPI.get_instance().size()*self.args.record_length
        self.summary['data_size_per_host_GB'] = data_per_node/1024./1024./1024.
        if DLIOMPI.get_instance().rank() == 0:
            logging.info(f"Total amount of data each host will consume is {data_per_node/1024./1024./1024} GB; each host has {self.summary['host_memory_GB']} GB memory") 
        if self.summary['data_size_per_host_GB'] <= self.output['host_memory_GB']:
            self.output['potential_caching'] = True
            if DLIOMPI.get_instance().rank() == 0: 
                logging.warning("The amount of dataset is smaller than the host memory; data might be cached after the first epoch. Increase the size of dataset to eliminate the caching effect!!!")
        potential_caching = []
        for i in range(DLIOMPI.get_instance().size()//DLIOMPI.get_instance().npernode()):
            if self.summary['host_memory_GB'][i]  <= self.summary['data_size_per_host_GB']:
                potential_caching.append(0)
            else:
                potential_caching.append(1)
        self.summary['potential_caching'] = potential_caching

    def start_run(self):
        self.start_run_timestamp = time()
    def end_run(self):
        self.end_run_timestamp = time()
        if not self.args.generate_only:
            total_elapsed_time = self.end_run_timestamp - self.start_run_timestamp
            train_au = np.array(self.comm.allreduce(np.array(self.train_au)))/self.comm.size
            train_throughput = self.comm.allreduce(np.array(self.train_throughput))
            self.summary['epochs'] = len(train_au)
            self.summary['metric']['train_au_percentage'] = list(train_au)
            self.summary['metric']['train_au_mean_percentage'] = np.mean(train_au)
            if self.summary['metric']['train_au_mean_percentage'] >=90:
                self.summary['metric']['train_au_meet_expectation'] = 'success'
            else:
                self.summary['metric']['train_au_meet_expectation'] = 'fail'
            self.summary['metric']['train_au_stdev_percentage'] = np.std(train_au)
            self.summary['metric']['train_throughput_samples_per_second'] = list(train_throughput)
            self.summary['metric']['train_throughput_mean_samples_per_second'] = np.mean(train_throughput)
            self.summary['metric']['train_throughput_stdev_samples_per_second'] = np.std(train_throughput)
            self.summary['metric']['train_io_mean_MB_per_second'] = np.mean(train_throughput)*self.record_size/1024./1024.
            self.summary['metric']['train_io_stdev_MB_per_second'] = np.std(train_throughput)*self.record_size/1024./1024.
            if self.args.do_eval:
                eval_au = np.array(self.comm.allreduce(self.eval_au))/self.comm.size
                eval_throughput = self.comm.allreduce(self.eval_throughput)
                self.summary['metric']['eval_au_percentage'] = list(eval_au)
                self.summary['metric']['eval_au_mean_percentage'] = np.mean(eval_au)
                if self.summary['metric']['eval_au_mean_percentage'] >=90:
                    self.summary['metric']['eval_au_meet_expectation'] = 'success'
                else:
                    self.summary['metric']['eval_au_meet_expectation'] = 'fail'
                self.summary['metric']['eval_au_stdev_percentage'] = np.std(eval_au)
                self.summary['metric']['eval_throughput_samples_per_second'] = list(eval_throughput)
                self.summary['metric']['eval_throughput_mean_samples_per_second'] = np.mean(eval_throughput)
                self.summary['metric']['eval_throughput_stdev_samples_per_second'] = np.std(eval_throughput)
                self.summary['metric']['eval_io_mean_MB_per_second'] = np.mean(eval_throughput)*self.record_size/1024./1024.
                self.summary['metric']['eval_io_stdev_MB_per_second'] = np.std(eval_throughput)*self.record_size/1024./1024.
            if self.my_rank==0:
                logging.info(f"{utcnow()} Saved outputs in {self.output_folder}")   
                metric="Averaged metric over all epochs\n[METRIC] ==========================================================\n"
                metric = metric + f"[METRIC] Number of Simulated Accelerators: {self.comm_size} \n"
                metric = metric + f"[METRIC] Training Accelerator Utilization [AU] (%): {np.mean(train_au):.4f} ({np.std(train_au):.4f})\n"
                metric = metric + f"[METRIC] Training Throughput (samples/second): {np.mean(train_throughput):.4f} ({np.std(train_throughput):.4f})\n"
                metric = metric + f"[METRIC] Training I/O Throughput (MB/second): {np.mean(train_throughput)*self.record_size/1024/1024:.4f} ({np.std(train_throughput)*self.record_size/1024/1024:.4f})\n"
                metric = metric + f"[METRIC] train_au_meet_expectation: {self.summary['metric']['train_au_meet_expectation']}\n"

                if self.args.do_eval:
                    metric = metric + f"[METRIC] Eval Accelerator Utilization [AU] (%): {np.mean(eval_au):.4f} ({np.std(eval_au):.4f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (samples/second): {np.mean(eval_throughput):.6f} ({np.std(eval_throughput):.6f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (MB/second): {np.mean(eval_throughput)*self.record_size/1024/1024:.6f} ({np.std(eval_throughput)*self.record_size/1024/1024:.6f})\n"
                    metric = metric + f"[METRIC] eval_au_meet_expectation: {self.summary['metric']['eval_au_meet_expectation']}\n"
                metric+="[METRIC] ==========================================================\n"
                logging.info(metric)   
    def start_train(self, epoch):   
        if self.my_rank == 0:
            ts = utcnow()
            if self.steps_override:
                logging.info(f"{ts} Starting epoch {epoch}: Overriding number of steps to {self.steps}.")
            else:
                logging.info(f"{ts} Starting epoch {epoch}: {self.steps} steps expected")
            self.per_epoch_stats[epoch] = {
                'start': ts,
            }
        # Initialize dicts for the current epoch
        self.output[epoch] = {}
        self.output[epoch]['load'] = {}
        self.output[epoch]['proc'] = {}
        self.output[epoch]['throughput'] = {}
        self.output[epoch]['au'] = {}
        self.output[epoch]['compute'] = {}

    def end_train(self, epoch, steps):
        au = np.array([self.output[epoch]['au'][k] for k in self.output[epoch]['au']])
        throughput = np.array([self.output[epoch]['throughput'][k] for k in self.output[epoch]['throughput']])
        steps = np.array([len(self.output[epoch]['proc'][k]) for k in self.output[epoch]['throughput']])
        if (np.sum(steps)==0):
            au = 0.0
            throughput = 0.0
        else:
            au = np.sum(au*steps)/np.sum(steps)
            throughput = np.sum(throughput*steps)/np.sum(steps)
        self.train_au.append(au)
        self.train_throughput.append(throughput)

        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch]['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            self.per_epoch_stats[epoch]['end'] = ts
            self.per_epoch_stats[epoch]['duration'] = duration
            logging.info(f"{ts} Ending epoch {epoch} - {np.sum(steps)} steps completed in {duration} s")

    def start_eval(self, epoch):
        self.start_timestamp = time()
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting eval - {self.steps_eval} steps expected")
            self.per_epoch_stats[epoch]['eval'] = {
                'start': ts
            }
        self.output[epoch]['load']['eval'] = []
        self.output[epoch]['proc']['eval'] = []
        self.output[epoch]['compute']['eval'] = []
        self.output[epoch]['au']['eval'] = 0.0
        self.output[epoch]['throughput']['eval'] = 0.0
    def end_eval(self, epoch):
        self.end_timestamp = time()
        self.compute_metrics_eval(epoch)
        self.eval_au.append(self.output[epoch]['au']['eval'])
        self.eval_throughput.append(self.output[epoch]['throughput']['eval'] )
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts)- pd.to_datetime(self.per_epoch_stats[epoch]['eval']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending eval - {self.steps_eval} steps completed in {duration} s")
            self.per_epoch_stats[epoch]['eval']['end'] = ts
            self.per_epoch_stats[epoch]['eval']['duration'] = duration        
            logging.info(f"{utcnow()} Epoch {epoch} [Eval] Accelerator Utilization [AU] (%): {self.output[epoch]['au']['eval']:.4f}")
            logging.info(f"{utcnow()} Epoch {epoch} [Eval] Throughput (samples/second): {self.output[epoch]['throughput']['eval']*self.comm_size:.4f}")

    def start_block(self, epoch, block):
        self.start_timestamp = time()
        self.output[epoch]['load'][f'block{block}'] = []
        self.output[epoch]['proc'][f'block{block}'] = []
        self.output[epoch]['throughput'][f'block{block}'] = []
        self.output[epoch]['au'][f'block{block}'] = []
        self.output[epoch]['compute'][f'block{block}'] = []
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting block {block}")
            self.per_epoch_stats[epoch][f'block{block}'] = {
                'start': ts
            }

    def end_block(self, epoch, block, steps_taken):
        self.end_timestamp = time()
        self.compute_metrics_train(epoch, block)
        
        if self.my_rank == 0:
            # Block was possibly already ended. Need this to end blocks
            # still ongoing when data loader runs out of batches and
            # does not take one of the expected exits from the batch reading loop
            if 'end' in self.per_epoch_stats[epoch][f'block{block}']:
                return
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'block{block}']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending block {block} - {steps_taken} steps completed in {duration} s")
            self.per_epoch_stats[epoch][f'block{block}']['end'] = ts
            self.per_epoch_stats[epoch][f'block{block}']['duration'] = duration
            logging.info(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Accelerator Utilization [AU] (%): {self.output[epoch]['au'][f'block{block}']:.4f}")
            logging.info(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Throughput (samples/second): {self.output[epoch]['throughput'][f'block{block}']*self.comm_size:.4f}")

    def start_ckpt(self, epoch, block, steps_taken):
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting checkpoint {block} after total step {steps_taken} for epoch {epoch}")
            self.per_epoch_stats[epoch][f'ckpt{block}'] = {
                'start': ts
            }

    def end_ckpt(self, epoch, block):
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'ckpt{block}']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending checkpoint {block} for epoch {epoch}")

            self.per_epoch_stats[epoch][f'ckpt{block}']['end'] = ts
            self.per_epoch_stats[epoch][f'ckpt{block}']['duration'] = duration

    def batch_loaded(self, epoch, step, block, t0):
        duration = time() - t0
        key = f'block{block}'
        if key in self.output[epoch]['load']:
            self.output[epoch]['load'][key].append(duration)
        else:
            self.output[epoch]['load'][key] = [duration]
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step}: loaded {self.batch_size} samples in {duration} s")


    def batch_processed(self, epoch, step, block, t0, computation_time):
        duration = time() - t0
        key = f'block{block}'
        if key in self.output[epoch]['proc']:
            self.output[epoch]['proc'][key].append(duration)
            self.output[epoch]['compute'][key].append(computation_time)
        else:
            self.output[epoch]['proc'] = [duration]
            self.output[epoch]['compute']=[computation_time]
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size} samples in {duration} s")

    def compute_metrics_train(self, epoch, block):
        key = f"block{block}"
        total_compute_time = np.sum(self.output[epoch]['compute'][key][1:])
        if (total_compute_time==0):
            au=0.0
        else:
            total_time = self.end_timestamp - self.start_timestamp - self.output[epoch]['proc'][key][0]
            au = total_compute_time / total_time
        throughput = len(self.output[epoch]['compute'][key])/(self.end_timestamp - self.start_timestamp)*self.batch_size
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def compute_metrics_eval(self, epoch):
        key = 'eval'
        total_compute_time = np.sum(self.output[epoch]['compute'][key][1:])
        if (total_compute_time==0):
            au=0.0
        else:
            total_time = self.end_timestamp - self.start_timestamp - self.output[epoch]['proc'][key][0]
            au = total_compute_time / total_time
        throughput = len(self.output[epoch]['compute'][key])/(self.end_timestamp - self.start_timestamp)*self.batch_size_eval
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def eval_batch_loaded(self, epoch, step, t0):
        duration = time() - t0
        self.output[epoch]['load']['eval'].append(duration)
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step} loaded {self.batch_size_eval} samples in {duration} s")


    def eval_batch_processed(self, epoch, step, t0, computation_time):
        duration = time() - t0
        self.output[epoch]['proc']['eval'].append(duration)
        self.output[epoch]['compute']['eval'].append(computation_time)
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size_eval} samples in {duration} s")
    def finalize(self):
        self.summary['end'] = utcnow()
    def save_data(self):
        # Dump statistic counters to files for postprocessing
        # Overall stats
        if self.my_rank == 0:
            with open(os.path.join(self.output_folder, 'per_epoch_stats.json'), 'w') as outfile:
                json.dump(self.per_epoch_stats, outfile, indent=4)
                outfile.flush()
            with open(os.path.join(self.output_folder, 'summary.json'), 'w') as outfile:
                json.dump(self.summary, outfile, indent=4)
        self.output['hostname'] = socket.gethostname()
        with open(os.path.join(self.output_folder, f'{self.my_rank}_output.json'), 'w') as outfile:
            json.dump(self.output, outfile, indent=4)
            outfile.flush()
        if self.my_rank == 0:
            logging.info(f"{utcnow()} outputs saved in RANKID_output.json")


