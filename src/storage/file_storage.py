from abc import ABC, abstractmethod
from time import time

from src.common.constants import MODULE_STORAGE
from src.storage.storage_handler import DataStorage, Namespace
from src.common.enumerations import NamespaceType, MetadataType
import os
import glob
import shutil

from src.utils.utility import event_logging, Profile


class FileStorage(DataStorage):
    """
    Storage APIs for creating files.
    """
    def __init__(self, namespace, framework=None):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_STORAGE):
            super().__init__(framework)
            self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)

    @event_logging(module=MODULE_STORAGE)
    def get_uri(self, id):
        return os.path.join(self.namespace.name, id)

    # Namespace APIs
    @event_logging(module=MODULE_STORAGE)
    def create_namespace(self, exist_ok=False):
        os.makedirs(self.namespace.name, exist_ok=exist_ok)
        return True

    @event_logging(module=MODULE_STORAGE)
    def get_namespace(self):
        return self.namespace.name

    # Metadata APIs
    @event_logging(module=MODULE_STORAGE)
    def create_node(self, id, exist_ok=False):
        os.makedirs(self.get_uri(id), exist_ok=exist_ok)
        return True

    @event_logging(module=MODULE_STORAGE)
    def get_node(self, id=""):
        path = self.get_uri(id)
        if os.path.exists(path):
            if os.path.isdir(path):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @event_logging(module=MODULE_STORAGE)
    def walk_node(self, id, use_pattern=False):
        if not use_pattern:
            return os.listdir(self.get_uri(id))
        else:
            return glob.glob(self.get_uri(id))

    @event_logging(module=MODULE_STORAGE)
    def delete_node(self, id):
        shutil.rmtree(self.get_uri(id))
        return True

    # TODO Handle partial read and writes
    @event_logging(module=MODULE_STORAGE)
    def put_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "w") as fd:
            fd.write(data)

    @event_logging(module=MODULE_STORAGE)
    def get_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "r") as fd:
            data = fd.read()
        return data

