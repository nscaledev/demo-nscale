"""
File service client for uploading files to Nscale.
"""

import requests
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

    def download_file(
        self,
        file_id: str,
        output_path: str,
        download_url: str = "",
        verbose: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Download a file to local disk.

        The fine-tuning file API provides file metadata via `get_file()`.
        If `download_url` is not passed, this method will look for a
        `download_url` field in that metadata.

        Args:
            file_id: File ID
            output_path: Destination path for downloaded file
            download_url: Optional direct URL to file contents
            verbose: Whether to print progress messages
            chunk_size: Stream chunk size in bytes

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If no download URL is available
            requests.HTTPError: If download request fails
        """
        if not download_url:
            metadata = self.get_file(file_id)
            download_url = metadata.get("download_url", "")

        if not download_url:
            raise ValueError(
                "No download URL available for this file. "
                "Pass `download_url` explicitly or ensure the API returns "
                "`download_url` in file metadata."
            )

        if verbose:
            print(f"Downloading file {file_id}...")

        headers = {
            "Authorization": self.client.session.headers.get("Authorization", "")
        }
        response = requests.get(
            download_url,
            headers=headers,
            stream=True,
            timeout=self.client.timeout,
        )
        response.raise_for_status()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        if verbose:
            print(f"✓ Downloaded file to: {output}")

        return str(output)
