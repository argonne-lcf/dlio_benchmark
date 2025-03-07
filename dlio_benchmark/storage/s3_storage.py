"""
   Copyright (c) 2024, UChicago Argonne, LLC
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
import boto3
from botocore.exceptions import ClientError

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class S3PytorchStorage(DataStorage):
    """
    PyTorch Storage APIs for creating files.
    It uses Boto3 client to read and write data
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        self.s3_client =  boto3.client('s3')


    @dlp.log
    def get_uri(self, id):
        return id        

    @dlp.log
    def create_namespace(self, exist_ok=False):
        # Assume the S3 bucket is exist
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
        return self.list_objects(self.namespace.name, id)

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        self.s3_client.put_object(Bucket=self.namespace.name, Key=id, Body=data.getvalue())
        return None

    @dlp.log
    def get_data(self, id, offset=None, length=None):
        obj_name = os.path.relpath(id)
        if offset:
            byte_range = f"bytes={offset}-{offset + length - 1}"
            return self.s3_client.get_object(Bucket=self.namespace.name, Key=id, Range=byte_range)['Body'].read()
        else:
            return self.s3_client.get_object(Bucket=self.namespace.name, Key=obj_name)['Body'].read()


    @dlp.log
    def list_objects(self, bucket_name, prefix=None):
        params = {'Bucket': bucket_name}
        if prefix:
            params['Prefix'] = prefix
        paths = []
        try:
            ## Need to implement pagination
            response = self.s3_client.list_objects_v2(**params)

            if 'Contents' in response:
                for key in response['Contents']:
                    paths.append(key['Key'][len(prefix)+1:])
        except self.s3_client.exceptions.NoSuchBucket:
            print(f"Bucket '{bucket_name}' does not exist.")

        return paths


    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    def get_basename(self, id):
        return os.path.basename(id)


class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

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
        return super().put_data(self.get_uri(id), data, offset, length)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        return super().get_data(self.get_uri(id), data, offset, length)

    def get_basename(self, id):
        return os.path.basename(id)
