"""
File service client for uploading files to Nscale.
"""

from pathlib import Path
from typing import Dict
from .base import BaseService


class FileService(BaseService):
    """Client for file upload operations."""

    def upload(self, file_path: str, verbose: bool = True) -> Dict:
        """
        Upload a file to Nscale.

        Args:
            file_path: Path to the file to upload
            verbose: Whether to print progress messages

        Returns:
            Dictionary containing file ID and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If file format is invalid
            ServerError: If upload fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if verbose:
            print(f"Uploading {path.name}...")

        with open(file_path, "rb") as f:
            files = {"file": (path.name, f, "text/csv")}
            result = self.client.request(
                "POST",
                "files",
                files=files,
                use_org_base=True,
            )

        if verbose:
            file_id = result.get("id")
            print(f"✓ Uploaded successfully. File ID: {file_id}")

        return result

    def list_files(self, limit: int = 100) -> Dict:
        """
        List uploaded files.

        Args:
            limit: Maximum number of files to return

        Returns:
            Dictionary with list of files
        """
        return self.client.get("files", params={"limit": limit})

    def get_file(self, file_id: str) -> Dict:
        """
        Get file metadata.

        Args:
            file_id: File ID

        Returns:
            Dictionary with file metadata
        """
        return self.client.get(f"files/{file_id}")

    def delete_file(self, file_id: str) -> Dict:
        """
        Delete a file.

        Args:
            file_id: File ID

        Returns:
            Response dictionary
        """
        return self.client.delete(f"files/{file_id}")
