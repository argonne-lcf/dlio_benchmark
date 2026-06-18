"""
   Copyright (c) 2025, UChicago Argonne, LLC
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

import logging
import os
import io

try:
    import multistorageclient as msc
    MSC_AVAILABLE = True
except ImportError:
    MSC_AVAILABLE = False
    Path = None
    logging.warning(
        "Multi-Storage Client (MSC) not available. "
        "Install with: pip install multi-storage-client"
    )

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class MscStorage(DataStorage):
    """
    Storage backend using NVIDIA Multi-Storage Client (MSC).
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(None)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        self.client, self.storage_root = msc.resolve_storage_client(self.namespace.name)
    
    @dlp.log
    def get_uri(self, id):
        """
        Get the URI for an object.
        """
        return os.path.join(self.storage_root, id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """
        Create the namespace for the storage. It's a noop for MSC.
        """
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id):
        assert isinstance(id, str), "id must be a string"
        try:
            info = self.client.info(self.get_uri(id))
            if info.type == "file":
                return MetadataType.FILE
            elif info.type == "directory":
                return MetadataType.DIRECTORY
            else:
                return None 
        except Exception:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        assert isinstance(id, str), "id must be a string"
        p = self.get_uri(id)
        try:
            objects = []
            for entry in self.client.list(path=p):
                name = entry.key
                relative_name = name[len(p):]
                if relative_name.startswith("/"):
                    relative_name = relative_name[1:]
                objects.append(relative_name)
            return objects
        except Exception as e:
            logging.error(f"Error walking {id}: {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        assert isinstance(id, str), "id must be a string"
        try:
            self.client.delete(self.get_uri(id), recursive=True)
            return True
        except Exception as e:
            logging.error(f"Error deleting {id}: {e}")
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        assert isinstance(id, str), "id must be a string"
        try:
            self.client.upload_file(self.get_uri(id), io.BytesIO(data))
            return True
        except Exception as e:
            logging.error(f"Error writing data to {id}: {e}")
            return False

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        assert isinstance(id, str), "id must be a string"
        p = self.get_uri(id)
        try:
            if offset is not None and length is not None:
                data = self.client.read_file(p, byte_range=msc.types.ByteRange(offset, length))
            else:
                data = self.client.read(p)
            return data
        except Exception as e:
            logging.error(f"Error reading data from {id}: {e}")
            return None

    @dlp.log
    def isfile(self, id):
        assert isinstance(id, str), "id must be a string"
        try:
            info = self.client.info(self.get_uri(id))
            return info.type == "file"
        except Exception:
            return False

    def get_basename(self, id):
        assert isinstance(id, str), "id must be a string"
        return os.path.basename(self.get_uri(id))

    def open(self, id):
        assert isinstance(id, str), "id must be a string"
        return self.client.open(self.get_uri(id))

    def upload_file(self, id, filename):
        assert isinstance(id, str), "id must be a string"
        assert isinstance(filename, str), "filename must be a string"
        self.client.upload_file(self.get_uri(id), filename)
