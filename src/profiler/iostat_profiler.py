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

from src.profiler.io_profiler import IOProfiler
import os
import signal
import subprocess as sp

class IostatProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if IostatProfiler.__instance is None:
            IostatProfiler()
        return IostatProfiler.__instance

    def __init__(self):
        super().__init__()
        self.my_rank = self._args.my_rank
        self.logfile = os.path.join(self._args.output_folder, 'iostat.json')
        """ Virtually private constructor. """
        if IostatProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            IostatProfiler.__instance = self

    def start(self):
        if self.my_rank == 0:
            # Open the logfile for writing
            self.logfile = open(self.logfile, 'w')
            #TODO: Get the relevant disks (from user input?)
            #iostat -c 5 -w 10 disk0
            #cmd = ["iostat", "-mdxtcy", "-o", "JSON", "sda", "sdb", "1"]
            cmd = ['iostat','-w', '1']
            self.process = sp.Popen(cmd, stdout=self.logfile, stderr=self.logfile)

    def stop(self):
        if self.my_rank == 0:
            self.logfile.flush()
            self.logfile.close()
            # If we send a stronger signal, the logfile json won't be ended correctly
            self.process.send_signal(signal.SIGINT) 
            # Might need a timeout here in case it hangs forever
            self.process.wait()