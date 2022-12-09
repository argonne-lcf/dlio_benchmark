from abc import ABC, abstractmethod
from src.storage.storage_handler import DataStorage, Namespace
from src.common.enumerations import NamespaceType, MetadataType
import os
import glob
import shutil

class FileStorage(DataStorage):
    """
    Storage APIs for creating files.
    """
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)

    def get_uri(self, id):
        return os.path.join(self.namespace.name, id)

    # Namespace APIs
    def create_namespace(self, exist_ok=False):
        os.makedirs(self.namespace.name, exist_ok=exist_ok)
        return True

    def get_namespace(self):
        return self.namespace.name

    # Metadata APIs
    def create_node(self, id, exist_ok=False):
        os.makedirs(self.get_uri(id), exist_ok=exist_ok)
        return True

    def get_node(self, id=""):
        path = self.get_uri(id)
        if os.path.exists(path):
            if os.path.isdir(path):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    def walk_node(self, id, use_pattern=False):
        if not use_pattern:
            return os.listdir(self.get_uri(id))
        else:
            return glob.glob(self.get_uri(id))

    def delete_node(self, id):
        shutil.rmtree(self.get_uri(id))
        return True

    # TODO Handle partial read and writes
    def put_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "w") as fd:
            fd.write(data)

    def get_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "r") as fd:
            data = fd.read()
        return data

