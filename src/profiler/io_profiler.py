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

from abc import ABC, abstractmethod

from src.utils.argument_parser import ArgumentParser


class IOProfiler(ABC):
    def __init__(self):
        self._arg_parser = ArgumentParser.get_instance()
        self.logdir = self._arg_parser.args.logdir

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass