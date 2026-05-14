"""
Live integration checks against Nscale APIs.

These tests are opt-in via marker:
    pytest -m integration
"""

import os
import pytest
import requests

from nscale import FineTuningClient, InferenceClient
from nscale.exceptions import (
    APIError,
    ResourceNotFoundError,
    ServerError,
    ValidationError,
)


def _resolve_credentials() -> tuple[str, str]:
    """
    Resolve API token and organization ID from environment.

    Returns:
        (token, organization_id)
    """
    token = os.getenv("NSCALE_TOKEN") or os.getenv("NSCALE_API_TOKEN")
    if not token:
        pytest.skip("NSCALE_TOKEN/NSCALE_API_TOKEN not set; skipping live integration test")

    org_id = os.getenv("ORGANIZATION_ID")
    if org_id:
        return token, org_id

    # Fallback: resolve organization from identity API.
    response = requests.get(
        "https://identity.nks.europe-west4.nscale.com/api/v1/acl",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    organizations = response.json().get("organizations", [])
    if not organizations:
        pytest.skip("No organizations available for this token; skipping live integration test")

    return token, organizations[0]["id"]


@pytest.mark.integration
def test_live_end_to_end_includes_download_clients(tmp_path):
    """
    Live smoke check that also exercises new download-related client APIs.
    """
    token, org_id = _resolve_credentials()

    # 1) Inference endpoint is reachable.
    with InferenceClient(api_token=token) as inference_client:
        models = inference_client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    # 2) Fine-tuning endpoints + new client APIs.
    with FineTuningClient(api_token=token, organization_id=org_id) as ft_client:
        base_models = ft_client.jobs.list_base_models(limit=10)
        assert isinstance(base_models, list)
        assert len(base_models) > 0

        # Existing endpoint context for available jobs.
        jobs = ft_client.jobs.list_jobs(limit=1)
        if jobs:
            job_id = jobs[0].get("id")
            if job_id:
                try:
                    result = ft_client.jobs.prepare_model_download(job_id)
                    assert isinstance(result, dict)
                except (ResourceNotFoundError, ValidationError, APIError, ServerError):
                    # Accepted: job may be non-completed or not eligible for download yet.
                    pass

        # New API 1: download_model path (prepare + error handling).
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        with pytest.raises((ResourceNotFoundError, ValidationError, APIError, ServerError)):
            ft_client.jobs.download_model(
                job_id=fake_job_id,
                output_path=str(tmp_path / "model.tar.gz"),
                verbose=False,
            )

        # New API 2: download_file path (metadata lookup + error handling).
        fake_file_id = "00000000-0000-0000-0000-000000000000"
        with pytest.raises((ResourceNotFoundError, ValidationError, APIError, ServerError)):
            ft_client.files.download_file(
                file_id=fake_file_id,
                output_path=str(tmp_path / "dataset.csv"),
                verbose=False,
            )
