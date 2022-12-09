from src.storage.file_storage import FileStorage
from src.storage.s3_storage import S3Storage
from src.common.enumerations import StorageType
from src.common.error_code import ErrorCodes

class StorageFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_storage(storage_type, namespace, framework=None):
        if storage_type == StorageType.LOCAL_FS:
            return FileStorage(namespace, framework)
        elif storage_type == StorageType.S3:
            return S3Storage(namespace, framework)
        else:
            raise Exception(str(ErrorCodes.EC1001))
