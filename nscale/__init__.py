"""
Nscale API Client Library

A Python client for interacting with Nscale's fine-tuning and inference APIs.

Example usage:
    from nscale import FineTuningClient, InferenceClient

    # Fine-tuning
    ft_client = FineTuningClient(
        api_token="your_token",
        organization_id="your_org_id"
    )

    dataset = ft_client.datasets.create(
        train_file_path="train.csv",
        validation_file_path="val.csv"
    )

    job = ft_client.jobs.create_job(
        dataset_id=dataset["id"],
        base_model_id="model_uuid",
        hyperparameters={...}
    )

    # Inference
    inf_client = InferenceClient(api_token="your_token")
    models = inf_client.list_models()
    answer = inf_client.generate_answer(
        model="Qwen/Qwen3-4B-Instruct-2507",
        question="What is AI?"
    )
"""

from .base import BaseService
from .platform import NscaleClient
from .files import FileService
from .datasets import DatasetService
from .finetuning import FineTuningService
from .inference import InferenceClient
from .exceptions import (
    NscaleError,
    AuthenticationError,
    ValidationError,
    ResourceNotFoundError,
    RateLimitError,
    ServerError,
    APIError,
)


class FineTuningClient:
    """
    High-level client for fine-tuning operations.

    Provides access to:
    - files: File upload service
    - datasets: Dataset management
    - jobs: Fine-tuning job management
    """

    def __init__(
        self,
        api_token: str,
        organization_id: str,
        base_url: str = "https://fine-tuning.api.nscale.com/api/v1",
        timeout: int = 120,
    ):
        """
        Initialize fine-tuning client.

        Args:
            api_token: Nscale API token
            organization_id: Organization ID
            base_url: Base URL for fine-tuning API
            timeout: Request timeout in seconds
        """
        # Create core client
        self._core = NscaleClient(
            api_token=api_token,
            organization_id=organization_id,
            base_url=base_url,
            timeout=timeout,
        )

        # Create service clients
        self.files = FileService(self._core)
        self.datasets = DatasetService(self._core, self.files)
        self.jobs = FineTuningService(self._core)

    def close(self):
        """Close the client session."""
        self._core.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = [
    "FineTuningClient",
    "InferenceClient",
    "NscaleClient",
    "BaseService",
    "FileService",
    "DatasetService",
    "FineTuningService",
    "NscaleError",
    "AuthenticationError",
    "ValidationError",
    "ResourceNotFoundError",
    "RateLimitError",
    "ServerError",
    "APIError",
]

__version__ = "0.1.0"
