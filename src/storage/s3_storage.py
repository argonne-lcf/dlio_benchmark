from time import time

from src.common.constants import MODULE_STORAGE
from src.storage.storage_handler import DataStorage, Namespace
from src.common.enumerations import NamespaceType, MetadataType
import os

from src.utils.utility import event_logging, Profile


class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    def __init__(self, namespace, framework=None):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_STORAGE):
            super().__init__(framework)
            self.namespace = Namespace(namespace, NamespaceType.FLAT)
    @event_logging(module=MODULE_STORAGE)
    def get_uri(self, id):
        return "s3://" + os.path.join(self.namespace.name, id)

    @event_logging(module=MODULE_STORAGE)
    def create_namespace(self, exist_ok=False):
        return True

    @event_logging(module=MODULE_STORAGE)
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @event_logging(module=MODULE_STORAGE)
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @event_logging(module=MODULE_STORAGE)
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @event_logging(module=MODULE_STORAGE)
    def walk_node(self, id, use_pattern=False):
        return super().walk_node(self.get_uri(id), use_pattern)

    @event_logging(module=MODULE_STORAGE)
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @event_logging(module=MODULE_STORAGE)
    def put_data(self, id, data, offset=None, length=None):
        return super().put_data(self.get_uri(id), data, offset, length)

    @event_logging(module=MODULE_STORAGE)
    def get_data(self, id, data, offset=None, length=None):
        return super().get_data(self.get_uri(id), data, offset, length)
