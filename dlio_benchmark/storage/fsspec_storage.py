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
from abc import ABC, abstractmethod
from time import time
import importlib
import os
import pdb

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.common.enumerations import FsspecPlugin
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.utility import Profile

try:
    import fsspec
except ImportError:
    fsspec = None

dlp = Profile(MODULE_STORAGE)

supported_fsspec_plugins = {
    FsspecPlugin.LOCAL_FS: {
                'type': NamespaceType.HIERARCHICAL,
                'plugin_name': 'file',
                'prefix': 'file://',
                'external_module': None
             }
}

class FsspecStorage(DataStorage):
    """
    Storage API for creating local filesystem files.
    """

# Different plugins will have different NamespaceType values.  Maybe we define
# a list at the top here of the plugins we support.  For each of them we define
# the prefix to use, e.g. "s3fs://".  For each of them there is also a list
# of the pip3 module that must be installed, which we test for and complain
# if missing.  We separately test for fsspec itself first.
#
# rehm: for daos, namespace will be <pool>:<cont>

    @dlp.log_init
    def __init__(self, namespace, fsspec_plugin, fsspec_extra_params):
        if fsspec is None:
            raise ModuleNotFoundError(
                "Package 'fsspec' must be installed in order to use fsspec-based storage types"
            )

        if not fsspec_plugin in supported_fsspec_plugins:
            raise Exception(
                "Unsupported fsspec plugin name '{0}' specified".format(fsspec_plugin)
            )

        plugin = supported_fsspec_plugins[fsspec_plugin]
        if plugin['external_module'] is not None:
            try:
                importlib.import_module(plugin['external_module'])
            except ImportError:
                raise ModuleNotFoundError(
                    "Package '{0}' must be installed in order to use fsspec plugin {1}".format(plugin['external_module'], fsspec_plugin)
                )

        self.fsspec_plugin = fsspec_plugin
        self.prefix = plugin['prefix']
        self.fs = fsspec.filesystem(plugin['plugin_name'])
# rehm: the namespace here must start with a / and must not end with a /.
# No, not true.  Do we check anything at all?  What about S3, doesn't a trailing
# slash mean something different?
        self.namespace = Namespace(namespace, plugin['type'])

    @dlp.log
    def get_uri(self, id=None):
        if id is None:
            return self.prefix + self.namespace.name
        else:
            return self.prefix + os.path.join(self.namespace.name, id)

# rehm: see https://snyk.io/advisor/python/fsspec/functions/fsspec.filesystem
# for the complexities of providing all the possible credentials.

    # Namespace APIs
    @dlp.log
    def create_namespace(self, exist_ok=False):
        pdb.set_trace()
        self.fs.makedirs(self.get_uri(), exist_ok=exist_ok)
        return True

    @dlp.log
    def get_namespace(self):
        pdb.set_trace()
        return self.namespace.name

    # Metadata APIs
    @dlp.log
    def create_node(self, id, exist_ok=False):
        pdb.set_trace()
        self.fs.makedirs(self.get_uri(id), exist_ok=exist_ok)
        return True

    @dlp.log
    def get_node(self, id=None):
        pdb.set_trace()
        path = self.get_uri(id)
        if self.fs.exists(path):
            if self.fs.isdir(path):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
# rehm: I need to test the NotFoundError logic here that exists for S3 and
# maybe duplicate it.  Is the exception the same?
        pdb.set_trace()
        if not use_pattern:
            return self.fs.listdir(self.get_uri(id))
        else:
            return self.fs.glob(self.get_uri(id))

    @dlp.log
    def delete_node(self, id):
        pdb.set_trace()
        self.fs.rm(self.get_uri(id), recursive=True)
        return True

    # TODO Handle partial read and writes
    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        pdb.set_trace()
        with self.fs.open(self.get_uri(id), mode="w") as fd:
            fd.write(data)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        pdb.set_trace()
        with self.fs.open(self.get_uri(id), mode="r") as fd:
            data = fd.read()
        return data

# rehm: get_flo() returns returns a file-like object that can be used in a
# with block.  Will this work as a parameeter to PIL.Image() and np()?  Does
# the FLO get cleaned up when it goes out of scope in the caller, the caller
# needs no close() function?  No, see:
#    https://filesystem-spec.readthedocs.io/en/latest/api.html
# at fsspec.core.OpenFile() which says that the caller must explicitly call
# close() when done to release the file descriptor.
# No, bigger problem, the caller won't have fsspec imported, can he call
# flo.close() then?
# What would happen if I returned flo.open() to the caller?  I can't call 
# flo.close() as that would affect the caller.  The flo will be lost after the
# return.  Maybe save in a dict, the fd points to the flo, and then provide
# a close() function with passes the fd?
# "as the low-level file object is not created until invoked with 'with'".
# Assuming it works, then partial read and write, seek, truncate, etc  will
# also work.
# Investigate adding compression and buffering options.

    @dlp.log
    def get_flo(self, id, mode="r"):
        pdb.set_trace()
        return self.fs.open(self.get_uri(id), mode=mode)
    
    def get_basename(self, id):
        pdb.set_trace()
        return os.path.basename(id)
