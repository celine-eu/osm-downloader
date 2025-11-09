import os
import boto3
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from osm_downloader.logger import get_logger

logger = get_logger(__name__)


def get_storage(cfg: Dict[str, Any]):
    """Initialize storage backend from config."""
    storage_cfg = cfg.get("storage", {})
    stype = os.getenv("STORAGE_TYPE", None) or storage_cfg.get("type", "fs")

    if stype == "fs":
        base_dir = os.getenv("CDS_DATA_DIR", None)
        if base_dir:
            logger.info(f"Using {base_dir} (CDS_DATA_DIR) for data storage")
        else:
            base_dir = storage_cfg.get("base_dir", None)
            if base_dir:
                logger.info(f"Using {base_dir} (config) for data storage")
            else:
                base_dir = "./data"
                logger.info(f"Using {base_dir} (default) for data storage")

        return FSStorage(base_dir=base_dir)

    elif stype == "s3":
        bucket = (
            os.getenv("AWS_BUCKET")
            or os.getenv("S3_BUCKET")
            or storage_cfg.get("bucket")
        )
        if not bucket:
            raise Exception(
                "`bucket` is missing. Add to config or via envs AWS_BUCKET or S3_BUCKET"
            )
        endpoint = storage_cfg.get("endpoint_url")
        return S3Storage(bucket=bucket, endpoint_url=endpoint)

    else:
        raise ValueError(f"Unknown storage type: {stype}")


class Storage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def save(self, local_path: str, key: str) -> None:
        pass

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def get_path(self, key: str) -> str:
        """Return a usable local path (download or direct)."""
        pass


class FSStorage(Storage):
    """Filesystem-based storage."""

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _full_path(self, key: str) -> str:
        return os.path.join(self.base_dir, key)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))

    def save(self, local_path: str, key: str) -> None:
        target = self._full_path(key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(local_path, target)  # atomic move

    def list(self, prefix: str = "") -> List[str]:
        results = []
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.base_dir)
                if rel.startswith(prefix):
                    results.append(rel)
        return results

    def get_path(self, key: str) -> str:
        return self._full_path(key)


class S3Storage(Storage):
    """S3/Minio-based storage."""

    def __init__(self, bucket: str, endpoint_url: str | None = None):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=(
                os.getenv("AWS_ENDPOINT_URL")
                or os.getenv("S3_ENDPOINT_URL")
                or endpoint_url
            ),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID")
            or os.getenv("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            or os.getenv("S3_SECRET_ACCESS_KEY"),
        )

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def save(self, local_path: str, key: str) -> None:
        self.s3.upload_file(local_path, self.bucket, key)

    def list(self, prefix: str = "") -> List[str]:
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def get_path(self, key: str) -> str:
        """Download to /tmp and return local path."""
        local_path = f"/tmp/{os.path.basename(key)}"
        self.s3.download_file(self.bucket, key, local_path)
        return local_path
