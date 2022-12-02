
from numpy import append
from src.utils.config import ConfigArguments
from src.utils.utility import utcnow

import os
import json
import math
import logging
import pandas as pd
from time import time

class StatsCounter(object):

    def __init__(self):
        self.args = ConfigArguments.get_instance()
        self.my_rank = self.args.my_rank
        self.output_folder = self.args.output_folder

        self.batch_size = self.args.batch_size
        self.batch_size_eval = self.args.batch_size_eval
        
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
        self.loading_times = {}
        self.processing_times = {}
        self.load_and_proc_times = {}

    def start_epoch(self, epoch):
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
        self.loading_times[epoch] = {}
        self.processing_times[epoch] = {}
        self.load_and_proc_times[epoch] = {}
        self.load_and_proc_times[epoch]['load'] = {}
        self.load_and_proc_times[epoch]['proc'] = {}

    def end_epoch(self, epoch, steps):
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch]['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            self.per_epoch_stats[epoch]['end'] = ts
            self.per_epoch_stats[epoch]['duration'] = duration
            logging.info(f"{ts} Ending epoch {epoch} - {steps} steps completed in {duration} s")

    def start_eval(self, epoch):
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting eval - {self.steps_eval} steps expected")
            self.per_epoch_stats[epoch]['eval'] = {
                'start': ts
            }
        self.load_and_proc_times[epoch]['load']['eval'] = []
        self.load_and_proc_times[epoch]['proc']['eval'] = []

    def end_eval(self, epoch):
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts)- pd.to_datetime(self.per_epoch_stats[epoch]['eval']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending eval - {self.steps_eval} steps completed in {duration} s")
            self.per_epoch_stats[epoch]['eval']['end'] = ts
            self.per_epoch_stats[epoch]['eval']['duration'] = duration        

    def start_block(self, epoch, block):
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting block {block}")
            self.per_epoch_stats[epoch][f'block{block}'] = {
                'start': ts
            }

    def end_block(self, epoch, block, steps_taken):
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
        if key in self.load_and_proc_times[epoch]['load']:
            self.load_and_proc_times[epoch]['load'][key].append(duration)
        else:
            self.load_and_proc_times[epoch]['load'][key] = [duration]
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step}: loaded {self.batch_size} samples in {duration} s")


    def batch_processed(self, epoch, step, block, t0):
        duration = time() - t0
        key = f'block{block}'
        if key in self.load_and_proc_times[epoch]['proc']:
            self.load_and_proc_times[epoch]['proc'][key].append(duration)
        else:
            self.load_and_proc_times[epoch]['proc'][key] = [duration]
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size} samples in {duration} s")


    def eval_batch_loaded(self, epoch, step, t0):
        duration = time() - t0
        self.load_and_proc_times[epoch]['load']['eval'].append(duration)
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step} loaded {self.batch_size_eval} samples in {duration} s")


    def eval_batch_processed(self, epoch, step, t0):
        duration = time() - t0
        self.load_and_proc_times[epoch]['proc']['eval'].append(duration)
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size_eval} samples in {duration} s")

    def save_data(self):
        # Dump statistic counters to files for postprocessing
        # Overall stats
        if self.my_rank == 0:
            with open(os.path.join(self.output_folder, 'per_epoch_stats.json'), 'w') as outfile:
                json.dump(self.per_epoch_stats, outfile, indent=4)

        with open(os.path.join(self.output_folder, f'{self.my_rank}_load_and_proc_times.json'), 'w') as outfile:
            json.dump(self.load_and_proc_times, outfile, indent=4)
