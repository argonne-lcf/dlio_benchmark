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
from time import time

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from urllib.parse import urlparse
import os
from s3torchconnector._s3client import S3Client, S3ClientConfig
import torch

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class S3PyTorchConnectorStorage(S3Storage):
    """
    Storage APIs for S3 objects.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}

        self.access_key_id = storage_options.get("access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        self.secret_access_key = storage_options.get("secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        self.endpoint = storage_options.get("endpoint_url", os.getenv("AWS_ENDPOINT_URL"))
        self.region = storage_options.get("region", os.getenv("AWS_REGION", "us-east-1"))

        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Build connector config, possibly with env overrides
        self.s3_client_config = S3ClientConfig(
            force_path_style=os.getenv("S3_FORCE_PATH_STYLE", "false").lower() == "true",
            max_attempts=int(os.getenv("S3_MAX_ATTEMPTS", "5")),
        )

        # Initialize the S3Client instance
        self.s3_client = S3Client(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

    @dlp.log
    def get_uri(self, id):
        return id

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        # Parse s3://bucket/prefix path
        parsed = urlparse(id)
        if parsed.scheme != 's3':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')

        if not use_pattern:
            return self.list_objects(bucket, prefix)
        else:
            ext = prefix.split('.')[-1]
            if ext != ext.lower():
                raise Exception(f"Unknown file format {ext}")

            # Pattern matching: check both lowercase and uppercase extensions
            lower_results = self.list_objects(bucket, prefix)
            upper_prefix = prefix.replace(ext, ext.upper())
            upper_results = self.list_objects(bucket, upper_prefix)

            return lower_results + upper_results

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        # Parse s3://bucket/prefix path
        parsed = urlparse(id)
        if parsed.scheme != 's3':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    
        bucket_name = parsed.netloc
        writer = self.s3_client.put_object(bucket_name, id)
        writer.write(data.getvalue())
        writer.close()
        return None

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        obj_name = id  # or just s3_key = id
        # Parse s3://bucket/prefix path
        parsed = urlparse(id)
        if parsed.scheme != 's3':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    
        bucket_name = parsed.netloc

        if offset is not None and length is not None:
            start = offset
            end = offset + length - 1
            reader = self.s3_client.get_object(bucket_name, obj_name, start=start, end=end)
        else:
            reader = self.s3_client.get_object(bucket_name, obj_name)

        return reader.read()        

    @dlp.log
    def list_objects(self, bucket_name, prefix=None):
        paths = []
        try:
            # list_objects returns an iterable stream of ObjectInfo
            prefix = f"s3://{bucket_name}/" + prefix.lstrip("/") + '/'
            obj_stream = self.s3_client.list_objects(bucket_name, prefix or "")

            for list_obj_result in obj_stream:
                for obj_info in list_obj_result.object_info:
                    key = obj_info.key
                    if prefix:
                        stripped_key = key[len(prefix):] if key.startswith(prefix) else key
                        paths.append(stripped_key)
                    else:
                        paths.append(key)
        except Exception as e:
            print(f"Error listing objects in bucket '{bucket_name}': {e}")

        return paths

    @dlp.log
    def isfile(self, id):
        return super().isfile(self.get_uri(id))

    def get_basename(self, id):
        return os.path.basename(id)
