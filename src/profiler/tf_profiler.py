"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""

from src.profiler.io_profiler import IOProfiler

class TFProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if TFProfiler.__instance is None:
            TFProfiler()
        return TFProfiler.__instance

    def __init__(self):
        super().__init__()
        """ Virtually private constructor. """
        if TFProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            TFProfiler.__instance = self

    def start(self):
        import tensorflow as tf
        tf.profiler.experimental.start(self.logdir)

    def stop(self):
        import tensorflow as tf
        tf.profiler.experimental.stop()