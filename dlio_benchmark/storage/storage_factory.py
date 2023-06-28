from dlio_benchmark.storage.file_storage import FileStorage
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.common.error_code import ErrorCodes

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
