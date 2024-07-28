# * coding utf-8 *
# @author: @github/sounishnath003
# createdAt: 25-07-2024

from google.auth import credentials
from google.cloud import storage
from loguru import logger


class Flywheel:

    def __init__(self, gcs_bucket: str, credentials_path: str) -> None:
        self.gcs_bucket = gcs_bucket
        self.credentials_path = gcs_bucket
        self.storage_client = storage.Client()

    def upload_dataset(self, filepath: str):
        """uploads the filepath directory which needs to be TXT file and to be uploaded in the storage bucket of gcp"""
        bucket = self.storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(filepath)
        blob.upload_from_filename(filepath)
        logger.info(f"File {filepath} uploaded to {self.gcs_bucket}.")
