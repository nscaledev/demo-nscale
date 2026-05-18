"""
Test suite for nscale.files module.

BDD Scenarios:
==============

Feature: File Upload Service
  As a developer working with Nscale
  I want to upload files for fine-tuning datasets
  So that I can create training datasets

  Scenario: Upload a file successfully
    Given a valid CSV file path
    When I call upload() with the file path
    Then the file should be uploaded via the platform client
    And the file metadata should be returned

  Scenario: Upload file with verbose output
    Given a valid file path and verbose=True
    When I call upload()
    Then progress messages should be printed
    And the file ID should be displayed

  Scenario: Upload non-existent file
    Given an invalid file path
    When I call upload()
    Then a FileNotFoundError should be raised

  Scenario: List uploaded files
    Given files exist in the system
    When I call list_files()
    Then it should return the list of files

  Scenario: Get file metadata
    Given a valid file ID
    When I call get_file()
    Then it should return the file metadata

  Scenario: Delete a file
    Given a valid file ID
    When I call delete_file()
    Then the file should be deleted via the API
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from nscale.files import FileService
from nscale.platform import NscaleClient


class TestFileServiceUpload:
    """Test file upload functionality."""

    def test_upload_file_successfully(self):
        """
        Scenario: Upload a file successfully
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.request.return_value = {
            "id": "file_123",
            "name": "test.csv",
            "bytes": 1024,
            "lines": 50
        }

        file_service = FileService(mock_client)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("question,answer\n")
            f.write("What is AI?,Artificial Intelligence\n")
            temp_path = f.name

        try:
            # Act
            result = file_service.upload(temp_path, verbose=False)

            # Assert
            mock_client.request.assert_called_once()
            call_args = mock_client.request.call_args

            assert call_args[0][0] == "POST"  # method
            assert call_args[0][1] == "files"  # endpoint
            assert "files" in call_args[1]
            assert call_args[1]["use_org_base"] is True

            assert result["id"] == "file_123"
            assert result["name"] == "test.csv"
        finally:
            os.unlink(temp_path)

    def test_upload_file_with_verbose_output(self, capsys):
        """
        Scenario: Upload file with verbose output
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.request.return_value = {
            "id": "file_456",
            "name": "data.csv"
        }

        file_service = FileService(mock_client)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n")
            temp_path = f.name

        try:
            # Act
            result = file_service.upload(temp_path, verbose=True)

            # Assert
            captured = capsys.readouterr()
            assert "Uploading" in captured.out
            assert "✓ Uploaded successfully" in captured.out
            assert "file_456" in captured.out
        finally:
            os.unlink(temp_path)

    def test_upload_nonexistent_file_raises_error(self):
        """
        Scenario: Upload non-existent file
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        file_service = FileService(mock_client)
        non_existent_path = "/path/to/nonexistent/file.csv"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="File not found"):
            file_service.upload(non_existent_path)


class TestFileServiceOperations:
    """Test file listing, retrieval, and deletion."""

    def test_list_files(self):
        """
        Scenario: List uploaded files
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "data": [
                {"id": "file_1", "name": "train.csv"},
                {"id": "file_2", "name": "val.csv"}
            ]
        }

        file_service = FileService(mock_client)

        # Act
        result = file_service.list_files(limit=50)

        # Assert
        mock_client.get.assert_called_once_with(
            "files",
            params={"limit": 50}
        )
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "file_1"

    def test_get_file_metadata(self):
        """
        Scenario: Get file metadata
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "id": "file_123",
            "name": "test.csv",
            "bytes": 2048,
            "lines": 100
        }

        file_service = FileService(mock_client)
        file_id = "file_123"

        # Act
        result = file_service.get_file(file_id)

        # Assert
        mock_client.get.assert_called_once_with(f"files/{file_id}")
        assert result["id"] == file_id
        assert result["bytes"] == 2048

    def test_delete_file(self):
        """
        Scenario: Delete a file
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.delete.return_value = {"status": "deleted"}

        file_service = FileService(mock_client)
        file_id = "file_to_delete"

        # Act
        result = file_service.delete_file(file_id)

        # Assert
        mock_client.delete.assert_called_once_with(f"files/{file_id}")
        assert result["status"] == "deleted"

    @patch("nscale.files.requests.get")
    def test_download_file_with_explicit_url(self, mock_get, tmp_path):
        """
        Scenario: Download file with explicit URL
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.session = Mock()
        mock_client.session.headers = {"Authorization": "Bearer token"}
        mock_client.timeout = 120

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"abc", b"123"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        file_service = FileService(mock_client)
        output_path = tmp_path / "downloaded.csv"

        # Act
        result = file_service.download_file(
            file_id="file_1",
            output_path=str(output_path),
            download_url="https://example.com/file.csv",
            verbose=False,
        )

        # Assert
        mock_get.assert_called_once()
        assert output_path.read_bytes() == b"abc123"
        assert result == str(output_path)

    @patch("nscale.files.requests.get")
    def test_download_file_uses_metadata_download_url(self, mock_get, tmp_path):
        """
        Scenario: Download file using URL from file metadata
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.session = Mock()
        mock_client.session.headers = {"Authorization": "Bearer token"}
        mock_client.timeout = 120
        mock_client.get.return_value = {"id": "file_1", "download_url": "https://example.com/file.csv"}

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"hello"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        file_service = FileService(mock_client)
        output_path = tmp_path / "downloaded.csv"

        # Act
        result = file_service.download_file(
            file_id="file_1",
            output_path=str(output_path),
            verbose=False,
        )

        # Assert
        mock_client.get.assert_called_once_with("files/file_1")
        assert output_path.read_bytes() == b"hello"
        assert result == str(output_path)

    def test_download_file_raises_when_no_download_url(self, tmp_path):
        """
        Scenario: Download fails when no URL is available
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {"id": "file_1", "name": "train.csv"}

        file_service = FileService(mock_client)
        output_path = tmp_path / "downloaded.csv"

        # Act & Assert
        with pytest.raises(ValueError, match="No download URL available"):
            file_service.download_file(
                file_id="file_1",
                output_path=str(output_path),
                verbose=False,
            )
