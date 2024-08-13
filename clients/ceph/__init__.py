import logging
import os
from pathlib import Path
from typing import List

import s3fs
from fsspec.callbacks import TqdmCallback

logger = logging.getLogger(__name__)


class S3fsS3Client:
    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "",
    ):
        if region_name:
            client_kwargs = {"endpoint_url": endpoint_url, "region_name": region_name}
        else:
            client_kwargs = {"endpoint_url": endpoint_url}
        self.fs = s3fs.S3FileSystem(
            key=aws_access_key_id,
            secret=aws_secret_access_key,
            client_kwargs=client_kwargs,
        )

    @property
    def endpoint_url(self) -> str:
        return self.fs.client_kwargs["endpoint_url"]

    def upload_file(self, file_path: str, bucket_name: str, key: str) -> str:
        full_path = f"{bucket_name}/{key}"
        logger.info(f"⬆️ Uploading {file_path} to {full_path}")
        with TqdmCallback(tqdm_kwargs={"unit": "B", "unit_scale": True}) as callback:
            self.fs.put_file(file_path, full_path, callback=callback)
        return f"{self.fs.client_kwargs['endpoint_url']}/{full_path}"

    def upload_directory(
        self, directory_path: str, bucket_name: str, prefix: str, verbose: bool = False
    ) -> str:
        for root, dirs, files in os.walk(directory_path):
            relative_path = os.path.relpath(root, directory_path)
            for file in files:
                local_file_path = os.path.join(root, file)
                if relative_path == ".":
                    s3_path = f"{prefix}/{file}"
                else:
                    s3_path = f"{prefix}/{relative_path}/{file}"
                if verbose:
                    print(f"Uploading {local_file_path} to {bucket_name}/{s3_path}")
                self.upload_file(local_file_path, bucket_name, s3_path)
        return f"{self.endpoint_url}/{bucket_name}/{prefix}"

    def download_file(self, bucket_name: str, key: str, file_path: str):
        full_path = f"{bucket_name}/{key}"
        logger.info(f"⬇️ Downloading {full_path} to {file_path}")
        with TqdmCallback(tqdm_kwargs={"unit": "B", "unit_scale": True}) as callback:
            self.fs.get_file(full_path, file_path, callback=callback)

    def download_directory(
        self, bucket_name: str, prefix: str, directory_path: str, recursive: bool = True
    ):
        full_path = f"{bucket_name}/{prefix}"
        if recursive:
            self.fs.get(full_path, directory_path, recursive=True)
        else:
            for file in self.fs.ls(full_path, detail=True):
                if file["type"] == "file":
                    self.fs.get(
                        file["name"],
                        os.path.join(directory_path, os.path.basename(file["name"])),
                    )
                else:
                    logger.warning(f"Skipping directory {file['name']} because recursive=False")

    def copy_file(
        self,
        source_bucket_name: str,
        source_key: str,
        target_bucket_name: str,
        target_key: str,
    ):
        source_path = f"{source_bucket_name}/{source_key}"
        target_path = f"{target_bucket_name}/{target_key}"
        self.fs.copy(source_path, target_path)

    def copy_directory_contents(
        self,
        source_bucket_name: str,
        source_prefix: str,
        target_bucket_name: str,
        target_prefix: str,
    ):
        source_path = f"{source_bucket_name}/{source_prefix}"
        target_path = f"{target_bucket_name}/{target_prefix}"
        for file in self.fs.ls(source_path, detail=True):
            if file["type"] == "file":
                source_key = str(file["name"])
                target_key = source_key.replace(source_path, target_path)
                self.fs.cp(source_key, target_key)
            else:
                logger.warning(f"Skipping directory {file['name']}")

    def move_file(
        self,
        source_bucket_name: str,
        source_key: str,
        target_bucket_name: str,
        target_key: str,
    ):
        source_path = f"{source_bucket_name}/{source_key}"
        target_path = f"{target_bucket_name}/{target_key}"
        self.fs.mv(source_path, target_path)

    def move_directory_contents(
        self,
        source_bucket_name: str,
        source_prefix: str,
        target_bucket_name: str,
        target_prefix: str,
    ):
        source_path = f"{source_bucket_name}/{source_prefix}"
        target_path = f"{target_bucket_name}/{target_prefix}"
        for file in self.fs.ls(source_path, detail=True):
            if file["type"] == "file":
                source_key = str(file["name"])
                target_key = source_key.replace(source_path, target_path)
                self.fs.cp(source_key, target_key)
            else:
                logger.warning(f"Skipping directory {file['name']}")

    def file_exists(self, bucket_name: str, key: str) -> bool:
        full_path = f"{bucket_name}/{key}"
        return self.fs.exists(full_path)

    def list_directory_contents(
        self, bucket_name: str, prefix: str = "", tail_only: bool = True
    ) -> List[str]:
        full_path = f"{bucket_name}/{prefix}"
        contents = self.fs.ls(full_path)
        if tail_only:
            return [os.path.basename(path) for path in contents]
        return contents

    def list_top_level_directories(
        self, bucket_name: str, prefix: str = "", tail_only: bool = True
    ) -> List[str]:
        full_path = f"{bucket_name}/{prefix}"
        dirs = self.fs.ls(full_path, detail=True)
        top_level_dirs = [d["name"] for d in dirs if d["type"] == "directory"]
        if tail_only:
            return [os.path.basename(d.rstrip("/")) for d in top_level_dirs]
        return top_level_dirs

    def list_top_level_files(
        self, bucket_name: str, prefix: str = "", tail_only: bool = True
    ) -> List[str]:
        full_path = f"{bucket_name}/{prefix}"
        files = self.fs.ls(full_path, detail=True)
        top_level_files = [f["name"] for f in files if f["type"] == "file"]
        if tail_only:
            return [os.path.basename(f) for f in top_level_files]
        return top_level_files

    def clear_directory_contents(self, bucket_name: str, prefix: str):
        full_path = f"{bucket_name}/{prefix}"
        if not self.fs.exists(full_path):  # directory may not exist
            return
        for file in self.fs.ls(full_path, detail=True):
            if file["type"] == "directory":
                bucket_name, prefix = file["name"].split("/", 1)
                self.clear_directory_contents(bucket_name=bucket_name, prefix=prefix)
                if self.fs.exists(file["name"]):
                    self.fs.rmdir(file["name"])
            else:
                self.fs.rm(file["name"])

    def delete_file(self, bucket_name: str, key: str):
        full_path = f"{bucket_name}/{key}"
        self.fs.rm(full_path)

    def delete_directory(self, bucket_name: str, prefix: str):
        self.clear_directory_contents(bucket_name, prefix)
        if self.fs.exists(f"{bucket_name}/{prefix}"):
            self.fs.rmdir(f"{bucket_name}/{prefix}")

    def create_bucket(self, bucket_name: str):
        self.fs.mkdir(bucket_name)

    def destroy_bucket(self, bucket_name: str, force: bool = False):
        if force:
            self.clear_directory_contents(bucket_name, "")
        self.fs.rmdir(bucket_name)

    def list_buckets(self) -> List[str]:
        buckets = [
            file["name"] for file in self.fs.ls("", detail=True) if file["type"] == "directory"
        ]
        return [os.path.basename(bucket) for bucket in buckets]

    def open(self, bucket_name: str, path: str, mode: str):
        return self.fs.open(str(Path(bucket_name) / path), mode)


class CephClient:
    def __init__(self, bucket: str = "ailab-research", common_prefix: str = "heap/"):
        self.bucket = bucket
        self.common_prefix = Path(common_prefix)
        self.s3_client = S3fsS3Client(
            endpoint_url="http://ceph.storage.buzzni.com",
            aws_access_key_id="17IVKU594D23KETUZZ5L",
            aws_secret_access_key="FmIc7L3Yeg2C5P5amIG3VqChvWyNvJ0AhpwWP3UB",
            region_name="ap-northeast-2",
        )

    def _prefix(self, path: str) -> str:
        return str(self.common_prefix / path)

    def _block_clobber(self, path: str):
        if self.s3_client.file_exists(self.bucket, self._prefix(path)):
            raise FileExistsError(f"{path} already exists")

    @property
    def base_location(self):
        return str(Path(self.bucket) / self.common_prefix)

    def ls(self, path: str = "") -> List[str]:
        return self.s3_client.list_directory_contents(self.bucket, self._prefix(path))

    def cp(self, src: str, dst: str, force: bool = False):
        if not force:
            self._block_clobber(dst)
        self.s3_client.copy_file(self.bucket, self._prefix(src), self.bucket, self._prefix(dst))

    def mv(self, src: str, dst: str, force: bool = False):
        if not force:
            self._block_clobber(dst)
        self.s3_client.copy_file(self.bucket, self._prefix(src), self.bucket, self._prefix(dst))

    def upload_file(self, src: str, dst: str, force: bool = False):
        if not force:
            self._block_clobber(dst)
        return self.s3_client.upload_file(src, self.bucket, self._prefix(dst))

    def upload_dir(self, src: str, dst: str, force: bool = False):
        if not force:
            self._block_clobber(dst)
        return self.s3_client.upload_directory(src, self.bucket, self._prefix(dst))

    def download_file(self, src: str, dst: str):
        self.s3_client.download_file(self.bucket, self._prefix(src), dst)

    def download_dir(self, src: str, dst: str):
        self.s3_client.download_directory(self.bucket, self._prefix(src), dst)

    def open(self, path: str, mode: str):
        return self.s3_client.open(self.bucket, self._prefix(path), mode)

    def rm(self, path: str):
        self.s3_client.delete_file(self.bucket, self._prefix(path))
