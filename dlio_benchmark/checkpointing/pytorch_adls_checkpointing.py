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
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import torch
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT

dlp = Profile(MODULE_CHECKPOINT)

# Import BlobIO at module level to allow test patching
try:
    from azstoragetorch.io import BlobIO
except ImportError:
    BlobIO = None

try:
    from azure.storage.blob import ContainerSasPermissions, generate_container_sas
except ImportError:
    ContainerSasPermissions = None
    generate_container_sas = None

class PyTorchADLSCheckpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchADLSCheckpointing.__instance is None:
            PyTorchADLSCheckpointing.__instance = PyTorchADLSCheckpointing()
        return PyTorchADLSCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "ptadls")

        # Check if BlobIO is available
        if BlobIO is None:
            raise ImportError(
                "azstoragetorch is required for ADLS Gen2 checkpointing support. "
                "Install with: pip install 'azstoragetorch>=0.1.0'"
            )

        # Access config values from self.args (inherited from BaseCheckpointing)
        storage_options = getattr(self.args, "storage_options", {}) or {}
        self._checkpoint_folder = self.args.checkpoint_folder
        self._account_name = None
        self._account_key = None
        self._shared_access_signature = None
        self._blobio_credential = None
        self._container_sas_tokens = {}
        self._container_sas_ttl = timedelta(hours=1)
        self._container_sas_refresh_margin = timedelta(minutes=5)

        if not isinstance(storage_options, dict):
            storage_options = dict(storage_options)

        # Support both connection string and account URL authentication
        connection_string = storage_options.get("connection_string")
        account_url = storage_options.get("account_url")
        account_name = storage_options.get("account_name")

        if connection_string:
            # Parse connection string and use SAS-based blob URLs with BlobIO
            self._load_connection_string(connection_string)
            self._blobio_credential = False
        elif account_url:
            # Use account URL with DefaultAzureCredential via BlobIO
            self._account_name = self._extract_account_name_from_url(account_url)
        elif account_name:
            # Use account name with DefaultAzureCredential via BlobIO
            self._account_name = account_name
        else:
            raise ValueError(
                "ADLS Gen2 checkpointing requires authentication configuration. "
                "Provide 'connection_string', 'account_url', or 'account_name' in storage_options."
            )

        if self._account_name is None:
            self._account_name = self._extract_account_name_from_abfs(self._checkpoint_folder)

        if self._account_name is None:
            raise ValueError(
                "Unable to determine ADLS account name for checkpointing. "
                "Provide storage_options.account_name/account_url or use canonical ABFS checkpoint URI."
            )

    def _load_connection_string(self, connection_string):
        parts = {}
        for segment in connection_string.split(';'):
            if '=' in segment:
                key, value = segment.split('=', 1)
                parts[key] = value

        self._account_name = parts.get("AccountName")
        self._account_key = parts.get("AccountKey")
        self._shared_access_signature = parts.get("SharedAccessSignature")

    def _extract_account_name_from_url(self, account_url):
        parsed = urlparse(account_url)
        host = parsed.netloc
        if not host:
            return None
        return host.split('.')[0]

    def _extract_account_name_from_abfs(self, uri):
        parsed = urlparse(uri)
        if parsed.scheme != "abfs" or '@' not in parsed.netloc:
            return None
        _, account_fqdn = parsed.netloc.split('@', 1)
        return account_fqdn.split('.')[0]

    def _to_blob_url(self, checkpoint_name, for_write):
        parsed = urlparse(checkpoint_name)

        if parsed.scheme == "https":
            blob_url = checkpoint_name
        elif parsed.scheme == "abfs":
            if '@' not in parsed.netloc:
                raise ValueError(
                    "Invalid ABFS checkpoint path. Expected format: "
                    "abfs://<file_system>@<account>.dfs.core.windows.net/<path>"
                )
            file_system, account_fqdn = parsed.netloc.split('@', 1)
            account_name = account_fqdn.split('.')[0]
            blob_path = parsed.path.lstrip('/')
            blob_url = f"https://{account_name}.blob.core.windows.net/{file_system}/{blob_path}"
        else:
            raise ValueError(
                f"Unsupported checkpoint URI '{checkpoint_name}'. Expected abfs:// or https://"
            )

        if self._shared_access_signature:
            return self._append_query(blob_url, self._shared_access_signature)

        if self._account_key:
            if generate_container_sas is None or ContainerSasPermissions is None:
                raise ImportError(
                    "azure-storage-blob is required for connection-string-based ADLS checkpointing."
                )
            blob_parsed = urlparse(blob_url)
            path_parts = blob_parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid blob URL for checkpointing: {blob_url}")
            container_name, _ = path_parts
            token = self._get_container_sas(container_name)
            return self._append_query(blob_url, token)

        return blob_url

    def _get_container_sas(self, container_name):
        cache_entry = self._container_sas_tokens.get(container_name)
        now = datetime.now(timezone.utc)
        refresh_margin = getattr(self, "_container_sas_refresh_margin", timedelta(minutes=5))

        if isinstance(cache_entry, dict):
            token = cache_entry.get("token")
            expires_at = cache_entry.get("expires_at")
            if token and expires_at and (expires_at - now) > refresh_margin:
                return token

        ttl = getattr(self, "_container_sas_ttl", timedelta(hours=1))
        expiry = now + ttl

        token = generate_container_sas(
            account_name=self._account_name,
            container_name=container_name,
            account_key=self._account_key,
            permission=ContainerSasPermissions(
                read=True,
                write=True,
                create=True,
                add=True,
                list=True,
            ),
            expiry=expiry,
        )
        self._container_sas_tokens[container_name] = {
            "token": token,
            "expires_at": expiry,
        }
        return token

    def _append_query(self, url, query_string):
        parsed = urlparse(url)
        existing = parse_qs(parsed.query, keep_blank_values=True)
        incoming = parse_qs(query_string.lstrip('?'), keep_blank_values=True)
        for key, values in incoming.items():
            existing[key] = values
        merged_query = urlencode(existing, doseq=True)
        return urlunparse(parsed._replace(query=merged_query))

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        blob_url = self._to_blob_url(name, for_write=True)
        # Save checkpoint to ADLS using azstoragetorch BlobIO
        with BlobIO(blob_url, "wb", credential=self._blobio_credential) as writer:
            torch.save(state, writer)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        blob_url = self._to_blob_url(name, for_write=False)
        state = dict() # clear up
        # Load checkpoint from ADLS using azstoragetorch BlobIO
        with BlobIO(blob_url, "rb", credential=self._blobio_credential) as reader:
            state = torch.load(reader)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert(len(state.keys())>0)

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()

