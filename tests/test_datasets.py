"""
Test suite for nscale.datasets module.

BDD Scenarios:
==============

Feature: Dataset Management Service
  As a developer preparing training data
  I want to create and manage datasets
  So that I can train models with my data

  Scenario: Create dataset successfully
    Given valid training and validation files
    When I call create() with the file paths
    Then both files should be uploaded
    And a dataset should be created with the file IDs
    And the dataset metadata should be returned

  Scenario: Create dataset with custom name
    Given file paths and a custom dataset name
    When I call create() with the name
    Then the dataset should be created with that name

  Scenario: Create dataset with auto-generated name
    Given file paths without a name
    When I call create()
    Then the dataset should be created with a timestamped name

  Scenario: Create dataset with verbose output
    Given file paths and verbose=True
    When I call create()
    Then progress messages should be displayed

  Scenario: List datasets
    Given datasets exist in the system
    When I call list_datasets()
    Then it should return the list of datasets

  Scenario: Get dataset details
    Given a valid dataset ID
    When I call get()
    Then it should return the dataset metadata

  Scenario: Delete dataset
    Given a valid dataset ID
    When I call delete()
    Then the dataset should be deleted
"""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime

from nscale.datasets import DatasetService
from nscale.platform import NscaleClient
from nscale.files import FileService


class TestDatasetServiceCreation:
    """Test dataset creation functionality."""

    def test_create_dataset_successfully(self):
        """
        Scenario: Create dataset successfully
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        # Mock file uploads
        mock_file_service.upload.side_effect = [
            {"id": "train_file_123", "name": "train.csv"},
            {"id": "val_file_456", "name": "val.csv"}
        ]

        # Mock dataset creation
        mock_client.post.return_value = {
            "id": "dataset_789",
            "name": "my_dataset",
            "training_file_id": "train_file_123",
            "validation_file_id": "val_file_456"
        }

        dataset_service = DatasetService(mock_client, mock_file_service)

        # Act
        result = dataset_service.create(
            train_file_path="data/train.csv",
            validation_file_path="data/val.csv",
            name="my_dataset",
            verbose=False
        )

        # Assert
        # Verify file uploads
        assert mock_file_service.upload.call_count == 2
        mock_file_service.upload.assert_any_call("data/train.csv", verbose=False)
        mock_file_service.upload.assert_any_call("data/val.csv", verbose=False)

        # Verify dataset creation
        mock_client.post.assert_called_once_with(
            "datasets",
            json={
                "name": "my_dataset",
                "training_file_id": "train_file_123",
                "validation_file_id": "val_file_456"
            }
        )

        assert result["id"] == "dataset_789"
        assert result["name"] == "my_dataset"

    def test_create_dataset_with_auto_generated_name(self):
        """
        Scenario: Create dataset with auto-generated name
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        mock_file_service.upload.side_effect = [
            {"id": "file1"},
            {"id": "file2"}
        ]

        mock_client.post.return_value = {"id": "dataset_123"}

        dataset_service = DatasetService(mock_client, mock_file_service)

        # Act
        with patch('nscale.datasets.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 1, 15, 10, 30, 45)
            mock_datetime.strftime = datetime.strftime

            result = dataset_service.create(
                train_file_path="train.csv",
                validation_file_path="val.csv",
                name=None,
                verbose=False
            )

        # Assert
        call_args = mock_client.post.call_args
        dataset_name = call_args[1]["json"]["name"]
        assert dataset_name.startswith("dataset_")
        assert "20260115" in dataset_name  # Check date format

    def test_create_dataset_with_verbose_output(self, capsys):
        """
        Scenario: Create dataset with verbose output
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        mock_file_service.upload.side_effect = [
            {"id": "file1"},
            {"id": "file2"}
        ]

        mock_client.post.return_value = {
            "id": "dataset_999",
            "name": "test_dataset"
        }

        dataset_service = DatasetService(mock_client, mock_file_service)

        # Act
        result = dataset_service.create(
            train_file_path="train.csv",
            validation_file_path="val.csv",
            verbose=True
        )

        # Assert
        captured = capsys.readouterr()
        assert "Creating dataset..." in captured.out
        assert "→ Uploading training file..." in captured.out
        assert "→ Uploading validation file..." in captured.out
        assert "→ Creating dataset configuration..." in captured.out
        assert "✓ Dataset created" in captured.out
        assert "dataset_999" in captured.out


class TestDatasetServiceOperations:
    """Test dataset listing, retrieval, and deletion."""

    def test_list_datasets(self):
        """
        Scenario: List datasets
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        mock_client.get.return_value = {
            "data": [
                {"id": "dataset_1", "name": "Dataset 1"},
                {"id": "dataset_2", "name": "Dataset 2"}
            ]
        }

        dataset_service = DatasetService(mock_client, mock_file_service)

        # Act
        result = dataset_service.list_datasets(limit=25)

        # Assert
        mock_client.get.assert_called_once_with(
            "datasets",
            params={"limit": 25}
        )
        assert len(result["data"]) == 2

    def test_get_dataset_details(self):
        """
        Scenario: Get dataset details
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        mock_client.get.return_value = {
            "id": "dataset_123",
            "name": "My Dataset",
            "training_file_id": "file1",
            "validation_file_id": "file2",
            "created_at": "2026-01-15T10:00:00Z"
        }

        dataset_service = DatasetService(mock_client, mock_file_service)
        dataset_id = "dataset_123"

        # Act
        result = dataset_service.get(dataset_id)

        # Assert
        mock_client.get.assert_called_once_with(f"datasets/{dataset_id}")
        assert result["id"] == dataset_id
        assert result["name"] == "My Dataset"

    def test_delete_dataset(self):
        """
        Scenario: Delete dataset
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_file_service = Mock(spec=FileService)

        mock_client.delete.return_value = {"status": "deleted"}

        dataset_service = DatasetService(mock_client, mock_file_service)
        dataset_id = "dataset_to_delete"

        # Act
        result = dataset_service.delete(dataset_id)

        # Assert
        mock_client.delete.assert_called_once_with(f"datasets/{dataset_id}")
        assert result["status"] == "deleted"
