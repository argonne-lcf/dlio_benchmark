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
from urllib.parse import urlparse

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os

from dlio_benchmark.utils.utility import Profile

# Import Azure SDK libraries at module level for patching in tests
try:
    from azure.storage.filedatalake import DataLakeServiceClient
    from azure.identity import DefaultAzureCredential
except ImportError:
    DataLakeServiceClient = None
    DefaultAzureCredential = None

dlp = Profile(MODULE_STORAGE)


class ADLSGen2Storage(DataStorage):
    """
    Storage APIs for ADLS Gen2 (Azure Data Lake Storage Gen2).
    Uses Azure Data Lake Storage Gen2 Python SDK to interact with Azure storage.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)
        self.container_name, self.account_fqdn, self.base_path = self._parse_namespace(namespace)
        
        # Check if Azure SDK libraries are available
        if DataLakeServiceClient is None:
            raise ImportError(
                "Azure Storage libraries are required for ADLS Gen2 support. "
                "Install with: pip install azure-storage-file-datalake azure-identity"
            )
        
        # Import exception types locally as they're only used in this class
        from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
        
        # Store exception types for use in methods
        self.ResourceNotFoundError = ResourceNotFoundError
        self.ResourceExistsError = ResourceExistsError
        
        # Get storage configuration from args
        storage_options = getattr(self._args, "storage_options", {}) or {}
        
        # Support both connection string and account URL authentication
        connection_string = storage_options.get("connection_string")
        account_url = storage_options.get("account_url")
        account_name = storage_options.get("account_name")

        if self.account_fqdn is None and account_name:
            self.account_fqdn = f"{account_name}.dfs.core.windows.net"
        elif self.account_fqdn is None and account_url:
            self.account_fqdn = urlparse(account_url).netloc
        
        if connection_string:
            # Use connection string authentication
            self.service_client = DataLakeServiceClient.from_connection_string(connection_string)
        elif account_url:
            # Use account URL with default credential
            credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(account_url=account_url, credential=credential)
        elif account_name:
            # Construct account URL from account name
            account_url = f"https://{account_name}.dfs.core.windows.net"
            credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError(
                "ADLS Gen2 requires authentication configuration. "
                "Provide 'connection_string', 'account_url', or 'account_name' in storage_options."
            )
        
        # Get or create file system client for the namespace (container)
        self.file_system_client = self.service_client.get_file_system_client(file_system=self.container_name)

    def _parse_namespace(self, namespace):
        parsed = urlparse(namespace)
        if parsed.scheme == "abfs":
            netloc = parsed.netloc
            if "@" not in netloc:
                raise ValueError(
                    "Invalid ABFS namespace URI. Expected format: "
                    "abfs://<file_system>@<account_name>.dfs.core.windows.net[/<path>]"
                )
            container_name, account_fqdn = netloc.split("@", 1)
            base_path = parsed.path.lstrip('/').rstrip('/')
            return container_name, account_fqdn, base_path

        namespace = namespace.strip('/')
        if '/' in namespace:
            container_name, base_path = namespace.split('/', 1)
            return container_name, None, base_path.strip('/')
        return namespace, None, ""

    def _build_abfs_uri(self, path):
        if self.account_fqdn:
            if path:
                return f"abfs://{self.container_name}@{self.account_fqdn}/{path}"
            return f"abfs://{self.container_name}@{self.account_fqdn}"

        if path:
            return f"abfs://{self.container_name}/{path}"
        return f"abfs://{self.container_name}"

    @dlp.log
    def get_uri(self, id):
        # If id is already a full URI, return as-is
        # Otherwise, construct the URI
        if id.startswith("abfs://"):
            return id
        path = self._resolve_path(id)
        return self._build_abfs_uri(path)

    def _resolve_path(self, uri):
        """
        Resolve URI or relative path to a path inside the ADLS container.
        """
        parsed = urlparse(uri)
        if parsed.scheme == 'abfs':
            if "@" not in parsed.netloc:
                raise ValueError(
                    "Invalid ABFS URI. Expected format: "
                    "abfs://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file_name>"
                )
            return parsed.path.lstrip('/')

        relative_path = uri.lstrip('/')
        if not self.base_path:
            return relative_path

        if not relative_path:
            return self.base_path

        if relative_path == self.base_path or relative_path.startswith(f"{self.base_path}/"):
            return relative_path

        return f"{self.base_path}/{relative_path}"

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """
        Create the file system (container) for ADLS Gen2.
        """
        try:
            self.file_system_client.create_file_system()
            return True
        except self.ResourceExistsError:
            if exist_ok:
                return True
            raise
        except Exception as e:
            print(f"Error creating namespace '{self.namespace.name}': {e}")
            return False

    @dlp.log
    def get_namespace(self):
        """
        Get the namespace (file system/container) information.
        """
        try:
            properties = self.file_system_client.get_file_system_properties()
            return MetadataType.DIRECTORY
        except self.ResourceNotFoundError:
            return None

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """
        Create a directory in ADLS Gen2.
        """
        try:
            dir_path = self._resolve_path(id)
            if not dir_path:
                return True
            directory_client = self.file_system_client.get_directory_client(dir_path)
            directory_client.create_directory()
            return True
        except self.ResourceExistsError:
            if exist_ok:
                return True
            raise
        except Exception as e:
            print(f"Error creating node '{id}': {e}")
            return False

    @dlp.log
    def get_node(self, id=""):
        """
        Get metadata about a path (file or directory).
        """
        if not id or id == "":
            return self.get_namespace()
        
        node_path = self._resolve_path(id)
        try:
            file_client = self.file_system_client.get_file_client(node_path)
            properties = file_client.get_file_properties()
            metadata = properties.get("metadata") or {}
            is_directory = str(metadata.get("hdi_isfolder", "")).lower() == "true"
            if is_directory:
                return MetadataType.DIRECTORY
            return MetadataType.FILE
        except self.ResourceNotFoundError:
            return None
        except Exception:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        """
        List files and directories under a path.
        """
        try:
            dir_path = self._resolve_path(id)
            if not use_pattern:
                # List all items in the directory
                paths = self.file_system_client.get_paths(path=dir_path, recursive=False)
                result = []
                prefix_len = len(dir_path.rstrip('/') + '/') if dir_path else 0
                
                for path in paths:
                    path_name = path.name
                    # Get only immediate children (not nested)
                    if prefix_len > 0:
                        relative_path = path_name[prefix_len:]
                    else:
                        relative_path = path_name
                    
                    # Only include immediate children (no slashes in relative path)
                    if '/' not in relative_path:
                        result.append(relative_path)
                
                return result
            else:
                # Pattern matching for file extensions
                format_ext = dir_path.split(".")[-1]
                if format_ext != format_ext.lower():
                    raise Exception(f"Unknown file format {format_ext}")
                
                search_path = os.path.dirname(dir_path)
                while any(token in search_path for token in ["*", "?", "["]):
                    search_path = os.path.dirname(search_path)

                # List files matching the pattern
                paths = self.file_system_client.get_paths(path=search_path)
                result = []
                
                # Match files with both lowercase and uppercase extensions
                lower_pattern = dir_path
                upper_pattern = dir_path.replace(format_ext, format_ext.upper())
                
                for path in paths:
                    path_name = path.name
                    if (path_name.endswith(format_ext) or 
                        path_name.endswith(format_ext.upper())):
                        result.append(self.get_uri(path_name))
                
                return result
        except Exception as e:
            print(f"Error walking node '{id}': {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        """
        Delete a file or directory from ADLS Gen2.
        """
        try:
            file_path = self._resolve_path(id)
            file_client = self.file_system_client.get_file_client(file_path)
            file_client.delete_file()
            return True
        except Exception as e:
            print(f"Error deleting node '{id}': {e}")
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """
        Upload data to a file in ADLS Gen2.
        """
        try:
            file_path = self._resolve_path(id)
            file_client = self.file_system_client.get_file_client(file_path)
            
            # Handle different data types
            if hasattr(data, 'getvalue'):
                # BytesIO or StringIO object
                data_bytes = data.getvalue()
            elif isinstance(data, bytes):
                data_bytes = data
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            if offset is not None and length is not None:
                # Partial write - append to existing file
                file_client.append_data(data_bytes, offset=offset, length=length)
                file_client.flush_data(offset + length)
            else:
                # Full write - create/overwrite file
                file_client.create_file()
                file_client.upload_data(data_bytes, overwrite=True)
            
            return True
        except Exception as e:
            print(f"Error putting data to '{id}': {e}")
            return False

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        """
        Download data from a file in ADLS Gen2.
        """
        try:
            file_path = self._resolve_path(id)
            file_client = self.file_system_client.get_file_client(file_path)
            
            if offset is not None and length is not None:
                # Partial read
                download_stream = file_client.download_file(offset=offset, length=length)
            else:
                # Full read
                download_stream = file_client.download_file()
            
            return download_stream.readall()
        except Exception as e:
            print(f"Error getting data from '{id}': {e}")
            return None

    @dlp.log
    def isfile(self, id):
        """
        Check if the path is a file.
        """
        try:
            file_path = self._resolve_path(id)
            file_client = self.file_system_client.get_file_client(file_path)
            properties = file_client.get_file_properties()
            metadata = properties.get("metadata") or {}
            is_directory = str(metadata.get("hdi_isfolder", "")).lower() == "true"
            return not is_directory
        except self.ResourceNotFoundError:
            return False
        except Exception:
            return False

    def get_basename(self, id):
        return os.path.basename(id)
