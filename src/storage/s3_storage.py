from src.storage.storage_handler import DataStorage, Namespace
from src.common.enumerations import NamespaceType, MetadataType
import os

class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

    def get_uri(self, id):
        return "s3://"+ os.path.join(self.namespace.name, id)

    def create_namespace(self, exist_ok=False):
        return True

    def get_namespace(self):
        return self.get_node(self.namespace.name)

    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    def walk_node(self, id, use_pattern=False):
        return super().walk_node(self.get_uri(id), use_pattern)

    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    def put_data(self, id, data, offset=None, length=None):
        return super().put_data(self.get_uri(id), data, offset, length)

    def get_data(self, id, data, offset=None, length=None):
        return super().get_data(self.get_uri(id), data, offset, length)
