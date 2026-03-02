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

import os
import io
import logging

try:
    from aistore.sdk import Client
    from aistore.sdk.bucket import Bucket
    from aistore.sdk.obj.object import Object
    from aistore.sdk.errors import AISError

    AISTORE_AVAILABLE = True
except ImportError:
    AISTORE_AVAILABLE = False
    # Define placeholders so mock.patch() can replace them in tests
    Client = None
    Bucket = None
    Object = None
    AISError = Exception
    logging.warning(
        "AIStore SDK not available. Install with: pip install aistore\n"
        "To use AIStore storage, set storage_type: aistore in your config."
    )

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class AIStoreStorage(DataStorage):
    """
    Native AIStore storage handler using the official AIStore Python SDK.
    This provides direct access to AIStore without going through S3 compatibility.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        if not AISTORE_AVAILABLE:
            raise ImportError(
                "AIStore SDK is required but not installed."
                "Install it with: `pip install aistore`"
            )

        # Call DataStorage.__init__ to set up framework
        super().__init__(framework)

        # Create namespace (AIStore uses flat namespace like S3)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}

        # AIStore endpoint (default: http://localhost:8080)
        self.endpoint = storage_options.get("endpoint_url", "http://localhost:8080")

        # Initialize AIStore client
        # Other parameters can be configured from environment variables
        self.client = Client(self.endpoint)

        # Bucket name from namespace
        self.bucket_name = self.namespace.name
        self.bucket = None

        logging.info(
            f"AIStore native storage initialized: endpoint={self.endpoint}, bucket=s3://{self.bucket_name}"
        )

    def _clean_key(self, id):
        """
        Extract the object key from a full S3/AIS URI.

        Why this is needed:
        - S3 generators (NPYGeneratorS3, NPYReaderS3) pass full URIs like:
          "s3://dlio-benchmark-native/train/img_08_of_16.npy"
          or "ais://dlio-benchmark-native/train/img_08_of_16.npy"
        - AIStore SDK expects just the object key:
          "train/img_08_of_16.npy"
        - This method strips the "s3://" or "ais://" prefix and bucket name

        Handles: 
          s3://bucket/path/file.ext -> path/file.ext
          ais://bucket/path/file.ext -> path/file.ext
        """
        key = str(id)

        # Remove all s3:// or ais:// prefixes (there might be multiple due to path construction)
        while key.startswith("s3://") or key.startswith("ais://"):
            if key.startswith("s3://"):
                key = key[5:]  # Remove "s3://"
            elif key.startswith("ais://"):
                key = key[6:]  # Remove "ais://"
            
            # After removing prefix, also remove bucket name if it's the next part
            if key.startswith(f"{self.bucket_name}/"):
                key = key[len(self.bucket_name) + 1 :]
            elif key.startswith(self.bucket_name):
                key = key[len(self.bucket_name) :]
                if key.startswith("/"):
                    key = key[1:]

        return key

    @dlp.log
    def get_uri(self, id):
        """
        Get the URI for an object.
        The data_folder config already includes ais://bucket, so just return id as-is.
        """
        return id

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """Create AIStore bucket if it doesn't exist"""
        self.bucket = self.client.bucket(self.bucket_name).create(exist_ok=exist_ok)
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """Create an object in AIStore"""
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        """Check if object exists"""
        try:
            if not self.bucket:
                self.bucket = self.client.bucket(self.bucket_name)

            key = self._clean_key(id) if id else ""

            if not key:  # Check bucket
                if self.bucket.head():
                    return {"type": "bucket"}
                return None

            # Check object
            obj = self.bucket.object(key)
            props = obj.head()
            if props:
                return {"type": "object"}
            return None
        except Exception as e:
            logging.debug(f"Object {id} not found: {e}")
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        """
        List objects with a given prefix.
        Returns just the filenames (relative to prefix) for DLIO compatibility.
        """
        try:
            if not self.bucket:
                self.bucket = self.client.bucket(self.bucket_name)

            prefix = self._clean_key(id) if id else ""
            objects = []

            # Use list_objects_iter for iterable results (not list_objects which returns BucketList)
            for entry in self.bucket.list_objects_iter(prefix=prefix):
                obj_name = entry.name

                # Remove the prefix to get just the filename
                # e.g., "train/img_00_of_16.npy" with prefix "train" -> "img_00_of_16.npy"
                if prefix and obj_name.startswith(prefix):
                    # Remove prefix
                    relative_name = obj_name[len(prefix) :]
                    # Remove leading slash if present
                    if relative_name.startswith("/"):
                        relative_name = relative_name[1:]
                    objects.append(relative_name)
                else:
                    objects.append(obj_name)

            logging.debug(f"walk_node: prefix={prefix}, found {len(objects)} objects")
            return objects
        except Exception as e:
            logging.error(f"Error walking node {id}: {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        """Delete an object from AIStore"""
        try:
            if not self.bucket:
                self.bucket = self.client.bucket(self.bucket_name)

            key = self._clean_key(id)
            obj = self.bucket.object(key)
            obj.delete()
            logging.debug(f"Deleted object: {key}")
            return True
        except Exception as e:
            logging.error(f"Error deleting node {id}: {e}")
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """Write data to AIStore object"""
        try:
            if not self.bucket:
                self.bucket = self.client.bucket(self.bucket_name)

            key = self._clean_key(id)

            # Convert data to bytes
            if isinstance(data, io.BytesIO):
                data.seek(0)
                body = data.read()
            elif isinstance(data, bytes):
                body = data
            else:
                body = bytes(data)

            # Put object
            obj = self.bucket.object(key)
            obj.get_writer().put_content(body)

            # TODO: add offset and length support

            logging.debug(f"Successfully uploaded: {key} ({len(body)} bytes)")
            return True
        except Exception as e:
            logging.error(f"Error putting data to {id}: {e}")
            return False

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        """Read data from AIStore object"""
        try:
            if not self.bucket:
                self.bucket = self.client.bucket(self.bucket_name)

            key = self._clean_key(id)
            obj = self.bucket.object(key)

            # Handle range reads
            byte_range = None
            if offset is not None and length is not None:
                # Both offset and length provided: "bytes=offset-(offset+length-1)"
                byte_range = f"bytes={offset}-{offset + length - 1}"
            elif offset is not None and length is None:
                # Only offset provided: "bytes=offset-"
                byte_range = f"bytes={offset}-"
            elif offset is None and length is not None:
                # Only length provided: "bytes=-length"
                byte_range = f"bytes=-{length}"

            if byte_range is not None:
                content = obj.get_reader(byte_range=byte_range).read_all()
            else:
                content = obj.get_reader().read_all()

            return content
        except Exception as e:
            logging.error(f"Error getting data from {id}: {e}")
            return None

    @dlp.log
    def isfile(self, id):
        """Check if object exists"""
        key = self._clean_key(id)
        obj = self.bucket.object(key)
        try:
            obj.head()
            return True
        except AISError:
            return False

    @dlp.log
    def get_basename(self, id):
        """Get the basename of a path"""
        return os.path.basename(id)
