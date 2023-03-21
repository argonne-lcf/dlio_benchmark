from time import time

from src.storage.storage_handler import DataStorage, Namespace
from src.common.enumerations import NamespaceType, MetadataType
import os

from src.utils.utility import PerfTrace,event_logging

MY_MODULE = "storage"


class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    def __init__(self, namespace, framework=None):
        t0 = time()
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0)

    @event_logging(module=MY_MODULE)
    def get_uri(self, id):
        return "s3://" + os.path.join(self.namespace.name, id)

    @event_logging(module=MY_MODULE)
    def create_namespace(self, exist_ok=False):
        return True

    @event_logging(module=MY_MODULE)
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @event_logging(module=MY_MODULE)
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @event_logging(module=MY_MODULE)
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @event_logging(module=MY_MODULE)
    def walk_node(self, id, use_pattern=False):
        return super().walk_node(self.get_uri(id), use_pattern)

    @event_logging(module=MY_MODULE)
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @event_logging(module=MY_MODULE)
    def put_data(self, id, data, offset=None, length=None):
        return super().put_data(self.get_uri(id), data, offset, length)

    @event_logging(module=MY_MODULE)
    def get_data(self, id, data, offset=None, length=None):
        return super().get_data(self.get_uri(id), data, offset, length)
