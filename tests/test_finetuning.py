"""
Test suite for nscale.finetuning module.

BDD Scenarios:
==============

Feature: Fine-Tuning Job Management
  As a developer training custom models
  I want to create and monitor fine-tuning jobs
  So that I can train models on my datasets

  Scenario: Create fine-tuning job successfully
    Given a valid dataset ID, model ID, and hyperparameters
    When I call create_job()
    Then a job should be created with the correct payload
    And the job metadata should be returned

  Scenario: Create job with auto-generated name
    Given job parameters without a name
    When I call create_job()
    Then the job should be created with a timestamped name

  Scenario: Handle nested API response structure
    Given the API returns nested job data
    When I call create_job()
    Then the job data should be extracted correctly from the wrapper

  Scenario: Create job fails with missing job ID
    Given the API returns a response without an ID
    When I call create_job()
    Then a ValueError should be raised

  Scenario: List fine-tuning jobs
    Given jobs exist in the system
    When I call list_jobs()
    Then it should return the list of jobs

  Scenario: List jobs with status filter
    Given I want only completed jobs
    When I call list_jobs(status="completed")
    Then only completed jobs should be returned

  Scenario: Get job status
    Given a valid job ID
    When I call get_job()
    Then it should return the job status and metadata

  Scenario: Get job metrics
    Given a job with training metrics
    When I call get_metrics()
    Then it should return the metrics data

  Scenario: Cancel running job
    Given a running job
    When I call cancel_job()
    Then the job should be cancelled

  Scenario: Monitor job until completion
    Given a job ID and it completes successfully
    When I call monitor_job()
    Then it should poll until completion
    And return the final status

  Scenario: Monitor job that fails
    Given a job that fails
    When I call monitor_job()
    Then it should detect the failure
    And return the failed status

  Scenario: List available base models
    Given base models exist
    When I call list_base_models()
    Then it should return the list of models
"""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime
import time

from nscale.finetuning import FineTuningService
from nscale.platform import NscaleClient
from nscale.exceptions import ServerError


class TestFineTuningJobCreation:
    """Test fine-tuning job creation."""

    def test_create_job_successfully(self):
        """
        Scenario: Create fine-tuning job successfully
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.return_value = {
            "job": {
                "id": "job_123",
                "name": "my_job",
                "status": "queued",
                "created_at": "2026-01-15T10:00:00Z"
            },
            "estimated_usage": {"tokens": 1000000}
        }

        service = FineTuningService(mock_client)

        hyperparameters = {
            "batch_size": 32,
            "learning_rate": 0.00001,
            "n_epochs": 3,
            "lora": {"enabled": True, "alpha": 16, "r": 8}
        }

        # Act
        result = service.create_job(
            dataset_id="dataset_456",
            base_model_id="model_789",
            hyperparameters=hyperparameters,
            prompt_column="question",
            answer_column="answer",
            name="my_job",
            verbose=False
        )

        # Assert
        mock_client.post.assert_called_once_with(
            "jobs",
            json={
                "name": "my_job",
                "base_model_id": "model_789",
                "dataset": {
                    "id": "dataset_456",
                    "prompt_column": "question",
                    "answer_column": "answer"
                },
                "hyperparameters": hyperparameters
            }
        )

        assert result["id"] == "job_123"
        assert result["status"] == "queued"

    def test_create_job_with_auto_generated_name(self):
        """
        Scenario: Create job with auto-generated name
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.return_value = {
            "job": {"id": "job_999", "name": "auto_generated"}
        }

        service = FineTuningService(mock_client)

        # Act
        with patch('nscale.finetuning.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 1, 15, 14, 30, 0)
            mock_datetime.strftime = datetime.strftime

            result = service.create_job(
                dataset_id="dataset_1",
                base_model_id="model_1",
                hyperparameters={"batch_size": 32},
                name=None,
                verbose=False
            )

        # Assert
        call_args = mock_client.post.call_args
        job_name = call_args[1]["json"]["name"]
        assert job_name.startswith("finetune_job_")
        assert "20260115" in job_name

    def test_create_job_handles_nested_response(self):
        """
        Scenario: Handle nested API response structure
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.return_value = {
            "job": {
                "id": "nested_job_id",
                "status": "starting"
            },
            "dry_run": False
        }

        service = FineTuningService(mock_client)

        # Act
        result = service.create_job(
            dataset_id="ds1",
            base_model_id="mdl1",
            hyperparameters={},
            verbose=False
        )

        # Assert
        assert result["id"] == "nested_job_id"
        assert "dry_run" not in result  # Should return job data, not wrapper

    def test_create_job_raises_error_on_missing_id(self):
        """
        Scenario: Create job fails with missing job ID
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.return_value = {
            "job": {"status": "queued"}  # No ID!
        }

        service = FineTuningService(mock_client)

        # Act & Assert
        with pytest.raises(ValueError, match="missing 'id' field"):
            service.create_job(
                dataset_id="ds1",
                base_model_id="mdl1",
                hyperparameters={},
                verbose=False
            )

    def test_create_job_handles_server_error(self):
        """
        Scenario: Create job encounters server error
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.side_effect = ServerError("Server error", status_code=500)

        service = FineTuningService(mock_client)

        # Act & Assert
        with pytest.raises(ServerError):
            service.create_job(
                dataset_id="ds1",
                base_model_id="mdl1",
                hyperparameters={},
                verbose=True
            )


class TestFineTuningJobOperations:
    """Test job listing, retrieval, and cancellation."""

    def test_list_jobs(self):
        """
        Scenario: List fine-tuning jobs
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "data": [
                {"id": "job_1", "status": "completed"},
                {"id": "job_2", "status": "running"}
            ]
        }

        service = FineTuningService(mock_client)

        # Act
        result = service.list_jobs(limit=10)

        # Assert
        mock_client.get.assert_called_once_with(
            "jobs",
            params={"limit": 10, "pagination": False}
        )
        assert len(result) == 2

    def test_list_jobs_with_status_filter(self):
        """
        Scenario: List jobs with status filter
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "data": [{"id": "job_1", "status": "completed"}]
        }

        service = FineTuningService(mock_client)

        # Act
        result = service.list_jobs(limit=5, status="completed")

        # Assert
        mock_client.get.assert_called_once_with(
            "jobs",
            params={"limit": 5, "pagination": False, "status": "completed"}
        )

    def test_get_job_status(self):
        """
        Scenario: Get job status
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "id": "job_123",
            "status": "running",
            "progress": 0.45
        }

        service = FineTuningService(mock_client)

        # Act
        result = service.get_job("job_123")

        # Assert
        mock_client.get.assert_called_once_with("jobs/job_123")
        assert result["status"] == "running"

    def test_get_job_metrics(self):
        """
        Scenario: Get job metrics
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = [
            {"step": 100, "train_loss": 0.5, "eval_loss": 0.6},
            {"step": 200, "train_loss": 0.4, "eval_loss": 0.5}
        ]

        service = FineTuningService(mock_client)

        # Act
        result = service.get_metrics("job_123")

        # Assert
        mock_client.get.assert_called_once_with("jobs/job_123/metrics")
        assert len(result) == 2
        assert result[0]["train_loss"] == 0.5

    def test_print_metrics_with_empty_list(self, capsys):
        """
        Scenario: Handle empty metrics list gracefully
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.organization_id = "test_org_id"
        mock_client.get.return_value = []  # Empty metrics

        service = FineTuningService(mock_client)

        # Act
        service._print_metrics("job_123")

        # Assert
        captured = capsys.readouterr()
        assert "Not available yet" in captured.out
        assert "console.nscale.com/organization/test_org_id/fine-tuning/jobs/job_123/overview" in captured.out

    def test_print_metrics_with_error(self, capsys):
        """
        Scenario: Handle metrics fetch error gracefully
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.organization_id = "test_org_id"
        mock_client.get.side_effect = Exception("Connection error")

        service = FineTuningService(mock_client)

        # Act
        service._print_metrics("job_123")

        # Assert
        captured = capsys.readouterr()
        assert "Could not fetch" in captured.out
        assert "Connection error" in captured.out
        assert "console.nscale.com/organization/test_org_id/fine-tuning/jobs/job_123/overview" in captured.out

    def test_cancel_job(self):
        """
        Scenario: Cancel running job
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.post.return_value = {"status": "cancelled"}

        service = FineTuningService(mock_client)

        # Act
        result = service.cancel_job("job_to_cancel")

        # Assert
        mock_client.post.assert_called_once_with("jobs/job_to_cancel/cancel")
        assert result["status"] == "cancelled"


class TestFineTuningJobMonitoring:
    """Test job monitoring functionality."""

    @patch('nscale.finetuning.time.sleep')
    def test_monitor_job_until_completion(self, mock_sleep):
        """
        Scenario: Monitor job until completion
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)

        # Simulate job progressing through states
        mock_client.get.side_effect = [
            {"status": "queued"},
            {"status": "starting"},
            {"status": "running"},
            {"status": "completed", "fine_tuned_model": "model_123"}
        ]

        service = FineTuningService(mock_client)

        # Act
        result = service.monitor_job("job_123", poll_interval=1, verbose=False)

        # Assert
        assert mock_client.get.call_count == 4
        assert result["status"] == "completed"
        assert mock_sleep.call_count == 3  # Sleeps between polls

    @patch('nscale.finetuning.time.sleep')
    def test_monitor_job_that_fails(self, mock_sleep):
        """
        Scenario: Monitor job that fails
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)

        mock_client.get.side_effect = [
            {"status": "running"},
            {"status": "failed", "error": "Out of memory"}
        ]

        service = FineTuningService(mock_client)

        # Act
        result = service.monitor_job("job_456", poll_interval=1, verbose=False)

        # Assert
        assert result["status"] == "failed"
        assert "error" in result


class TestFineTuningBaseModels:
    """Test listing base models."""

    def test_list_base_models(self):
        """
        Scenario: List available base models
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {
            "data": [
                {"id": "model_1", "name": "Llama-3.1-8B"},
                {"id": "model_2", "name": "Mistral-7B"}
            ]
        }

        service = FineTuningService(mock_client)

        # Act
        result = service.list_base_models(limit=50)

        # Assert
        mock_client.get.assert_called_once_with(
            "base-models",
            params={"limit": 50}
        )
        assert len(result) == 2
        assert result[0]["name"] == "Llama-3.1-8B"


class TestFineTuningModelDownload:
    """Test fine-tuned model download flows."""

    def test_prepare_model_download(self):
        """
        Scenario: Prepare model download URL
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {"download_url": "https://example.com/model.tar.gz"}
        service = FineTuningService(mock_client)

        # Act
        result = service.prepare_model_download("job_123")

        # Assert
        mock_client.get.assert_called_once_with("jobs/job_123/download")
        assert result["download_url"] == "https://example.com/model.tar.gz"

    def test_download_model_with_explicit_url(self, tmp_path):
        """
        Scenario: Download model with explicit URL
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.session = Mock()
        mock_client.session.headers = {"Authorization": "Bearer token"}
        mock_client.timeout = 120

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"tar", b"gz"]
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        service = FineTuningService(mock_client)
        output_path = tmp_path / "model.tar.gz"

        # Act
        result = service.download_model(
            job_id="job_123",
            output_path=str(output_path),
            download_url="https://example.com/model.tar.gz",
            verbose=False,
        )

        # Assert
        assert output_path.read_bytes() == b"targz"
        assert result == str(output_path)

    def test_download_model_uses_prepare_endpoint(self, tmp_path):
        """
        Scenario: Download model by first preparing URL
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {"download_url": "https://example.com/model.tar.gz"}
        mock_client.session = Mock()
        mock_client.session.headers = {"Authorization": "Bearer token"}
        mock_client.timeout = 120

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"model"]
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        service = FineTuningService(mock_client)
        output_path = tmp_path / "model.tar.gz"

        # Act
        result = service.download_model(
            job_id="job_123",
            output_path=str(output_path),
            verbose=False,
        )

        # Assert
        mock_client.get.assert_called_once_with("jobs/job_123/download")
        assert output_path.read_bytes() == b"model"
        assert result == str(output_path)

    def test_download_model_raises_when_no_download_url(self, tmp_path):
        """
        Scenario: Download fails when prepare endpoint has no URL
        """
        # Arrange
        mock_client = Mock(spec=NscaleClient)
        mock_client.get.return_value = {}
        service = FineTuningService(mock_client)
        output_path = tmp_path / "model.tar.gz"

        # Act & Assert
        with pytest.raises(ValueError, match="Model download URL not found"):
            service.download_model(
                job_id="job_123",
                output_path=str(output_path),
                verbose=False,
            )
