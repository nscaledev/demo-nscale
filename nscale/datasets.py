"""
Dataset service client for managing fine-tuning datasets.
"""

from datetime import datetime
from typing import Dict, Optional
from .base import BaseService
from .files import FileService


class DatasetService(BaseService):
    """Client for dataset operations."""

    def __init__(self, client, file_service: FileService):
        """
        Initialize dataset service.

        Args:
            client: Initialized NscaleClient instance
            file_service: FileService instance for uploads
        """
        super().__init__(client)
        self.file_service = file_service

    def create(
        self,
        train_file_path: str,
        validation_file_path: str,
        name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Create a dataset by uploading training and validation files.

        Args:
            train_file_path: Path to training CSV file
            validation_file_path: Path to validation CSV file
            name: Name for the dataset (auto-generated if not provided)
            verbose: Whether to print progress messages

        Returns:
            Dictionary containing dataset ID and metadata

        Raises:
            FileNotFoundError: If files don't exist
            ValidationError: If dataset creation fails
            ServerError: If API returns 500 error
        """
        if verbose:
            print("Creating dataset...")

        # Generate default name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dataset_{timestamp}"

        # Upload training file
        if verbose:
            print(f"  → Uploading training file...")
        train_file_resp = self.file_service.upload(train_file_path, verbose=False)

        # Upload validation file
        if verbose:
            print(f"  → Uploading validation file...")
        val_file_resp = self.file_service.upload(validation_file_path, verbose=False)

        # Create dataset configuration
        if verbose:
            print(f"  → Creating dataset configuration...")

        payload = {
            "name": name,
            "training_file_id": train_file_resp["id"],
            "validation_file_id": val_file_resp["id"],
        }

        result = self.client.post("datasets", json=payload)

        if verbose:
            dataset_id = result.get("id")
            print(f"✓ Dataset created. Dataset ID: {dataset_id}")

        return result

    def list_datasets(self, limit: int = 100) -> Dict:
        """
        List datasets.

        Args:
            limit: Maximum number of datasets to return

        Returns:
            Dictionary with list of datasets
        """
        return self.client.get("datasets", params={"limit": limit})

    def get(self, dataset_id: str) -> Dict:
        """
        Get dataset details.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dictionary with dataset metadata
        """
        return self.client.get(f"datasets/{dataset_id}")

    def delete(self, dataset_id: str) -> Dict:
        """
        Delete a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            Response dictionary
        """
        return self.client.delete(f"datasets/{dataset_id}")
