"""
Fine-tuning service client for managing training jobs.
"""

import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from tqdm import tqdm

from .base import BaseService
from .exceptions import ServerError


class FineTuningService(BaseService):
    """Client for fine-tuning job operations."""

    def create_job(
        self,
        dataset_id: str,
        base_model_id: str,
        hyperparameters: Dict,
        prompt_column: str = "question",
        answer_column: str = "answer",
        name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Create a fine-tuning job.

        Args:
            dataset_id: ID of the dataset to use
            base_model_id: Base model ID (UUID)
            hyperparameters: Training hyperparameters dict
            prompt_column: Name of the prompt column in dataset
            answer_column: Name of the answer column in dataset
            name: Name for the fine-tuning job (auto-generated if not provided)
            verbose: Whether to print progress messages

        Returns:
            Dictionary containing job ID and metadata

        Raises:
            ValidationError: If parameters are invalid
            ServerError: If API returns 500 error (known issue)
        """
        if verbose:
            print(f"Creating fine-tuning job for model: {base_model_id}...")

        # Generate default name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"finetune_job_{timestamp}"

        # Build payload matching PubMedQA cookbook format
        payload = {
            "name": name,
            "base_model_id": base_model_id,
            "dataset": {
                "id": dataset_id,
                "prompt_column": prompt_column,
                "answer_column": answer_column,
            },
            "hyperparameters": hyperparameters,
        }

        if verbose:
            print(f"Payload: {payload}")

        try:
            result = self.client.post("jobs", json=payload)
        except ServerError as e:
            # Special handling for known 500 error issue
            if verbose:
                print("\n" + "=" * 70)
                print("⚠️  500 INTERNAL SERVER ERROR")
                print("=" * 70)
                print("This appears to be an API-side issue with Nscale.")
                print()
                print("The request payload is valid (dataset and model IDs exist),")
                print("but the server is returning an internal error.")
                print()
                print("Next steps:")
                print("  1. Check Nscale console: https://console.nscale.com")
                print("  2. Try again in a few minutes (may be temporary)")
                print("  3. Contact Nscale support if issue persists")
                print("  4. Reference: ISSUE_REPORT.md for detailed diagnostics")
                print("=" * 70)
            raise

        # Parse response - API returns nested structure with 'job' key
        job_data = result.get("job")
        if not job_data:
            # Fallback: maybe API returns flat structure?
            job_data = result

        job_id = job_data.get("id")
        if not job_id:
            raise ValueError(
                f"API response missing 'id' field. Response: {result}"
            )

        if verbose:
            print(f"✓ Fine-tuning job created. Job ID: {job_id}")

        # Return the job object, not the wrapper
        return job_data

    def list_jobs(
        self,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """
        List fine-tuning jobs.

        Args:
            limit: Maximum number of jobs to return
            status: Filter by status (queued, starting, running, completed, failed, cancelled)

        Returns:
            List of jobs with their metadata
        """
        params = {"limit": limit, "pagination": False}
        if status:
            params["status"] = status

        result = self.client.get("jobs", params=params)
        return result.get("data", [])

    def get_job(self, job_id: str) -> Dict:
        """
        Get the status of a fine-tuning job.

        Args:
            job_id: ID of the job

        Returns:
            Dictionary containing job status and metadata
        """
        return self.client.get(f"jobs/{job_id}")

    def get_metrics(self, job_id: str) -> List[Dict]:
        """
        Get training metrics for a job.

        Args:
            job_id: ID of the job

        Returns:
            List of metric dictionaries
        """
        return self.client.get(f"jobs/{job_id}/metrics")

    def cancel_job(self, job_id: str) -> Dict:
        """
        Cancel a running job.

        Args:
            job_id: ID of the job

        Returns:
            Response dictionary
        """
        return self.client.post(f"jobs/{job_id}/cancel")

    def monitor_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        verbose: bool = True,
    ) -> Dict:
        """
        Monitor a fine-tuning job until completion.

        Args:
            job_id: ID of the job to monitor
            poll_interval: Seconds between status checks
            verbose: Whether to print progress

        Returns:
            Final job status
        """
        if verbose:
            print(f"\nMonitoring job {job_id}...")
            print(f"Console: https://console.nscale.com/jobs/{job_id}\n")

        with tqdm(desc="Training", unit="check", disable=not verbose) as pbar:
            while True:
                status = self.get_job(job_id)
                state = status.get("status", "unknown")

                if verbose:
                    pbar.set_postfix_str(f"Status: {state}")

                if state in ["succeeded", "completed"]:
                    if verbose:
                        print("\n✓ Training completed successfully!")
                        self._print_metrics(job_id)
                    return status
                elif state in ["failed", "cancelled"]:
                    if verbose:
                        print(f"\n✗ Training {state}")
                        if "error" in status:
                            print(f"Error: {status['error']}")
                    return status

                time.sleep(poll_interval)
                if verbose:
                    pbar.update(1)

    def _print_metrics(self, job_id: str) -> None:
        """Print training metrics for a job."""
        org_id = self.client.organization_id
        console_url = f"https://console.nscale.com/organization/{org_id}/fine-tuning/jobs/{job_id}/overview"

        try:
            metrics = self.get_metrics(job_id)
            if metrics and isinstance(metrics, list) and len(metrics) > 0:
                print("\nTraining Metrics:")
                latest = metrics[-1]
                print(f"  Train Loss: {latest.get('train_loss', 'N/A')}")
                print(f"  Eval Loss: {latest.get('eval_loss', 'N/A')}")
                print(f"  Perplexity: {latest.get('perplexity', 'N/A')}")
            else:
                print("\nTraining Metrics: Not available yet")
                print(f"  View in console: {console_url}")
        except Exception as e:
            # More informative error message
            error_msg = str(e)
            print("\nTraining Metrics: Could not fetch")
            if hasattr(e, 'status_code'):
                print(f"  Error: HTTP {e.status_code}")
            elif "404" in error_msg or "Not Found" in error_msg:
                print(f"  Note: Job may still be initializing")
            else:
                print(f"  Error: {type(e).__name__}: {error_msg}")
            print(f"  View in console: {console_url}")

    def list_base_models(self, limit: int = 100) -> List[Dict]:
        """
        List available base models for fine-tuning.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of available models with their IDs and metadata
        """
        result = self.client.get("base-models", params={"limit": limit})
        return result.get("data", [])

    def prepare_model_download(self, job_id: str) -> Dict:
        """
        Prepare download for a completed fine-tuning job.

        Args:
            job_id: ID of the completed fine-tuning job

        Returns:
            Dictionary with download metadata (typically includes `download_url`)
        """
        return self.client.get(f"jobs/{job_id}/download")

    def download_model(
        self,
        job_id: str,
        output_path: str,
        download_url: str = "",
        chunk_size: int = 1024 * 1024,
        verbose: bool = True,
    ) -> str:
        """
        Download fine-tuned model archive to local disk.

        Args:
            job_id: Fine-tuning job ID
            output_path: Destination path for `.tar.gz` archive
            download_url: Optional direct download URL. If omitted, this method
                calls `prepare_model_download(job_id)` to get one.
            chunk_size: Stream chunk size in bytes
            verbose: Whether to print progress messages

        Returns:
            Path to the downloaded archive

        Raises:
            ValueError: If no download URL is available
            requests.HTTPError: If download request fails
        """
        if not download_url:
            prepare_result = self.prepare_model_download(job_id)
            download_url = prepare_result.get("download_url", "")

        if not download_url:
            raise ValueError(
                "Model download URL not found in prepare response. "
                "Pass `download_url` explicitly or verify API response."
            )

        if verbose:
            print(f"Downloading model for job {job_id}...")

        # Reuse configured client session so retry policy applies to transient
        # backend/storage errors during artifact download.
        response = self.client.session.get(
            download_url,
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
            print(f"✓ Model downloaded to: {output}")

        return str(output)
