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
import logging
from dlio_benchmark.utils.utility import utcnow, DLIOMPI

from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.common.enumerations import FormatType, DataLoaderType, StorageType
from dlio_benchmark.common.error_code import ErrorCodes


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_reader(type, dataset_type, thread_index, epoch_number):
        """
        This function set the data reader based on the data format and the data loader specified. 
        """

        _args = ConfigArguments.get_instance()
        if _args.reader_class is not None:
            if DLIOMPI.get_instance().rank() == 0:
                self.logger.info(f"{utcnow()} Running DLIO with custom data loader class {_args.reader_class.__name__}")
            return _args.reader_class(dataset_type, thread_index, epoch_number)
        elif type == FormatType.HDF5:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            elif _args.storage_type in (StorageType.S3, StorageType.AISTORE):
                storage_library = (getattr(_args, "storage_options", {}) or {}).get("storage_library")
                if storage_library in ("s3dlio", "s3torchconnector", "minio"):
                    from dlio_benchmark.reader.hdf5_reader_s3_iterable import HDF5ReaderS3Iterable
                    return HDF5ReaderS3Iterable(dataset_type, thread_index, epoch_number)
                from dlio_benchmark.reader.hdf5_reader import HDF5Reader
                return HDF5Reader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.hdf5_reader import HDF5Reader
                return HDF5Reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.CSV:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            elif _args.storage_type in (StorageType.S3, StorageType.AISTORE):
                storage_library = (getattr(_args, "storage_options", {}) or {}).get("storage_library")
                if storage_library in ("s3dlio", "s3torchconnector", "minio"):
                    from dlio_benchmark.reader.csv_reader_s3_iterable import CSVReaderS3Iterable
                    return CSVReaderS3Iterable(dataset_type, thread_index, epoch_number)
                from dlio_benchmark.reader.csv_reader import CSVReader
                return CSVReader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.csv_reader import CSVReader
                return CSVReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.JPEG or type == FormatType.PNG:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            elif _args.data_loader == DataLoaderType.NATIVE_DALI:
                from dlio_benchmark.reader.dali_image_reader import DaliImageReader
                return DaliImageReader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader import create_image_reader
                return create_image_reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.NPY:
            if _args.data_loader == DataLoaderType.NATIVE_DALI:
                from dlio_benchmark.reader.dali_npy_reader import DaliNPYReader
                return DaliNPYReader(dataset_type, thread_index, epoch_number)
            elif _args.odirect == True:
                from dlio_benchmark.reader.npy_reader_odirect import NPYReaderODirect
                return NPYReaderODirect(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader import create_npy_reader
                return create_npy_reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.NPZ:
            if _args.data_loader == DataLoaderType.NATIVE_DALI:
                raise Exception("Loading data of %s format is not supported without framework data loader; please use npy format instead." %type)
            elif _args.odirect == True:
                from dlio_benchmark.reader.npz_reader_odirect import NPZReaderODIRECT
                return NPZReaderODIRECT(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader import create_npz_reader
                return create_npz_reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.TFRECORD:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            elif (getattr(_args, "storage_options", {}) or {}).get("storage_library") == "s3dlio":
                # s3dlio handles both s3:// and file:// URIs.
                from dlio_benchmark.reader.tfrecord_reader_s3_iterable import TFRecordReaderS3Iterable
                return TFRecordReaderS3Iterable(dataset_type, thread_index, epoch_number)
            if _args.data_loader == DataLoaderType.NATIVE_DALI:
                from dlio_benchmark.reader.dali_tfrecord_reader import DaliTFRecordReader
                return DaliTFRecordReader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.tf_reader import TFReader
                return TFReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.INDEXED_BINARY:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.indexed_binary_reader import IndexedBinaryReader
                return IndexedBinaryReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.MMAP_INDEXED_BINARY:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.indexed_binary_mmap_reader import IndexedBinaryMMapReader
                return IndexedBinaryMMapReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.SYNTHETIC:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.synthetic_reader import SyntheticReader
                return SyntheticReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.PARQUET:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            # s3dlio streaming loader: unified path for all URI schemes
            # (s3://, file://, direct://, az://, gs://).
            # Opt-in via storage_options.storage_library: s3dlio.
            # Two decode modes (set storage_options.decode):
            #   raw   (default) — no decode; pure I/O measurement
            #   arrow           — Rust Arrow IPC decode via create_async_loader
            storage_lib = (getattr(_args, "storage_options", {}) or {}).get("storage_library")
            decode = (getattr(_args, "storage_options", {}) or {}).get("decode", "raw")
            if storage_lib == "s3dlio":
                if decode == "arrow":
                    from dlio_benchmark.reader.parquet_reader_s3dlio_arrow import ParquetReaderS3dlioArrow
                    return ParquetReaderS3dlioArrow(dataset_type, thread_index, epoch_number)
                else:
                    from dlio_benchmark.reader.parquet_reader_s3dlio import ParquetReaderS3dlio
                    return ParquetReaderS3dlio(dataset_type, thread_index, epoch_number)
            elif _args.storage_type in (StorageType.S3, StorageType.AISTORE):
                from dlio_benchmark.reader.parquet_reader_s3_iterable import ParquetReaderS3Iterable
                return ParquetReaderS3Iterable(dataset_type, thread_index, epoch_number)
            elif _args.storage_type in (StorageType.LOCAL_FS,):
                # If storage_library=direct, reuse ParquetReaderS3Iterable with direct:// URIs
                # (s3dlio O_DIRECT reads — bypasses page cache, true parity with S3 path).
                # Fall back to ParquetReaderFileIterable for storage_library=posix (or unset).
                if storage_lib == "direct":
                    from dlio_benchmark.reader.parquet_reader_s3_iterable import ParquetReaderS3Iterable
                    return ParquetReaderS3Iterable(dataset_type, thread_index, epoch_number)
                else:
                    from dlio_benchmark.reader.parquet_reader_file_iterable import ParquetReaderFileIterable
                    return ParquetReaderFileIterable(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.parquet_reader import ParquetReader
                return ParquetReader(dataset_type, thread_index, epoch_number)


        else:
            raise Exception("Loading data of %s format is not supported without framework data loader" %type)
