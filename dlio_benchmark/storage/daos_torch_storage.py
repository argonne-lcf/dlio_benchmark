"""
   Copyright (c) 2026, Enakta Labs Ltd
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

import os

from pydaos.torch import Dataset
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.config import ConfigArguments
from concurrent.futures import ThreadPoolExecutor,  as_completed
from multiprocessing import cpu_count

dlp = Profile(MODULE_STORAGE)


class DaosTorchStorage(DataStorage):
    """
    This is very ad-hoc implementation of DataStorage interface for pydaos.torch evaluation.
    It supports only read only operation.
    There's no generic, POSIX like interface yet so this implementation relies only on what Dataset provides:
    list of file names. Which, then converted to list of files and directories so get_node and walk_node
    operate on these two lists.
    """

    @dlp.log_init
    def __init__(self, namespace="/", framework=None):
        super().__init__(framework)

        args = ConfigArguments.get_instance()

        dataset = Dataset(args.daos_pool, args.daos_cont, args.data_folder)
        files = [name for (name, size) in dataset.objects]

        def get_dir(fname):
            d = os.path.dirname(fname)
            return os.path.normpath(d)

        def process_chunk(chunk):
            return {get_dir(fname) for fname in chunk}

        workers = cpu_count()
        chunk_size = len(files) // min(workers, len(files))

        dirs = set()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_chunk, files[i:i + chunk_size])
                       for i in range(0, len(files), chunk_size)]

            for future in as_completed(futures):
                dirs.update(future.result())

        self._dirs = dirs
        self._files = files
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)

    @dlp.log
    def get_uri(self, id):
        return os.path.join(self.namespace.name, id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """
        This will only work for checkpoints: DAOS Checkpoint interface ensures that path exists
        before writing the checkpoint file.
        """
        return True

    @dlp.log
    def get_node(self, id=""):
        path = self.get_uri(id)
        path = os.path.normpath(path)

        for dir in self._dirs:
            if dir.startswith(path):
                return MetadataType.DIRECTORY
        return MetadataType.FILE

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        path = self.get_uri(id)

        if use_pattern:
            path = path[:path.find("*")]
            return [f for f in self._files if f.startswith(path)]

        if not path.endswith(os.sep):
            path += os.sep

        pref_len = len(path)
        files = [f for f in self._files if f.startswith(path) and f.find(os.sep, pref_len) < 0]
        dirs = [d for d in self._dirs if d.startswith(path) and d.find(os.sep, pref_len) < 0 and len(d) > pref_len]

        return files + dirs

    @dlp.log
    def delete_node(self, id):
        raise NotImplementedError

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        raise NotImplementedError

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        raise NotImplementedError

    def get_basename(self, id):
        return os.path.basename(id)
