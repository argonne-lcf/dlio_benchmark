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
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os
from s3torchconnector import S3Client, S3ClientConfig
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
        self.region = os.getenv("AWS_REGION", "us-east-1")

        # Build connector config, possibly with env overrides
        self.client_config = S3ClientConfig(
            force_path_style=os.getenv("S3_FORCE_PATH_STYLE", "false").lower() == "true",
            max_attempts=int(os.getenv("S3_MAX_ATTEMPTS", "5")),
        )

        # Initialize the S3Client instance
        self.client = S3Client(
            region=self.region,
            s3client_config=self.client_config,
        )        

    @dlp.log
    def get_uri(self, id):
        return "s3://" + os.path.join(self.namespace.name, id)

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
        return super().walk_node(self.get_uri(id), use_pattern)

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        writer = self.s3_client.put_object(self.bucket, key)
        writer.write(data)
        writer.commit()        
        return None

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        obj_name = os.path.relpath(id)  # or just s3_key = id

        if offset is not None and length is not None:
            start = offset
            end = offset + length - 1
            reader = self.s3_client.get_object(self.namespace.name, obj_name, start=start, end=end)
        else:
            reader = self.s3_client.get_object(self.namespace.name, obj_name)

        return reader.read()        

    @dlp.log
    def list_objects(self, bucket_name, prefix=None):
        paths = []
        try:
            # list_objects returns an iterable stream of ObjectInfo
            obj_stream = self.s3_client.list_objects(bucket_name, prefix or "")

            for obj_info in obj_stream:
                key = obj_info.key
                if prefix:
                    stripped_key = key[len(prefix)+1:] if key.startswith(prefix) else key
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
