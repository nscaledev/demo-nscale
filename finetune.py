#!/usr/bin/env python3
"""
Fine-tuning script for Nscale API.

This script handles the complete fine-tuning workflow:
1. Create a dataset (uploads training and validation CSV files)
2. Launch a fine-tuning job
3. Monitor training progress
"""

import os
import time
import requests
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv
from tqdm import tqdm
import sys


class NscaleClient:
    """Client for interacting with Nscale API."""

    def __init__(
        self,
        api_token: str,
        organization_id: str,
        base_url: str = "https://fine-tuning.api.nscale.com/api/v1"
    ):
        self.api_token = api_token
        self.organization_id = organization_id
        # Construct full base URL with organization ID
        self.base_url = f"{base_url.rstrip('/')}/organizations/{organization_id}"
        self.headers = {
            "Authorization": f"Bearer {api_token}"
        }

    def list_models(self) -> List[Dict]:
        """
        List available base models for fine-tuning.

        Returns:
            List of available models with their IDs and metadata
        """
        response = requests.get(
            f"{self.base_url}/base-models",
            headers=self.headers,
            params={"limit": 100}  # Get up to 100 models
        )
        response.raise_for_status()
        result = response.json()
        # Return the data array from the response
        return result.get("data", [])

    def upload_file(self, file_path: str, verbose: bool = True) -> Dict:
        """
        Upload a file to Nscale.

        Args:
            file_path: Path to the file to upload
            verbose: Whether to print progress messages

        Returns:
            Dictionary containing file ID and metadata
        """
        if verbose:
            print(f"Uploading {file_path}...")

        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "text/csv")}

            response = requests.post(
                f"{self.base_url}/files",
                headers={"Authorization": f"Bearer {self.api_token}"},
                files=files
            )

        response.raise_for_status()
        result = response.json()
        if verbose:
            print(f"✓ Uploaded successfully. File ID: {result.get('id')}")
        return result

    def create_dataset(
        self,
        train_file_path: str,
        validation_file_path: str,
        name: Optional[str] = None,
        prompt_column: str = "question",
        answer_column: str = "answer"
    ) -> Dict:
        """
        Create a dataset by uploading training and validation files.
        This internally handles both file uploads and dataset creation.

        Args:
            train_file_path: Path to training CSV file
            validation_file_path: Path to validation CSV file
            name: Name for the dataset (auto-generated if not provided)
            prompt_column: Name of the prompt column
            answer_column: Name of the answer column

        Returns:
            Dictionary containing dataset ID and metadata
        """
        print("Creating dataset...")

        # Generate a default name if not provided
        if name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dataset_{timestamp}"

        # Step 1a: Upload training file
        print(f"  → Uploading {Path(train_file_path).name}...")
        train_file_resp = self.upload_file(train_file_path, verbose=False)

        # Step 1b: Upload validation file
        print(f"  → Uploading {Path(validation_file_path).name}...")
        val_file_resp = self.upload_file(validation_file_path, verbose=False)

        # Step 1c: Create dataset configuration
        print(f"  → Creating dataset configuration...")
        # Note: prompt_column and answer_column go in job creation, not dataset creation
        payload = {
            "name": name,
            "training_file_id": train_file_resp["id"],
            "validation_file_id": val_file_resp["id"]
        }

        response = requests.post(
            f"{self.base_url}/datasets",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload
        )

        if response.status_code != 200:
            print(f"\n✗ Error creating dataset:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            print(f"Payload sent: {payload}")

        response.raise_for_status()
        result = response.json()
        print(f"✓ Dataset created. Dataset ID: {result.get('id')}")
        return result

    def create_finetune_job(
        self,
        dataset_id: str,
        model: str,
        hyperparameters: Dict,
        name: Optional[str] = None,
        prompt_column: str = "question",
        answer_column: str = "answer",
        lora_config: Optional[Dict] = None
    ) -> Dict:
        """
        Create a fine-tuning job.

        Args:
            dataset_id: ID of the dataset to use
            model: Base model to fine-tune
            hyperparameters: Training hyperparameters
            name: Name for the fine-tuning job (auto-generated if not provided)
            prompt_column: Name of the prompt column in dataset
            answer_column: Name of the answer column in dataset
            lora_config: LoRA configuration (optional)

        Returns:
            Dictionary containing job ID and metadata
        """
        print(f"Creating fine-tuning job for model: {model}...")

        # Generate a default name if not provided
        if name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"finetune_job_{timestamp}"

        # Add LoRA config to hyperparameters if provided
        if lora_config and lora_config.get("enabled"):
            hyperparameters["lora"] = {
                "enabled": True,
                "alpha": lora_config.get("alpha", 16),
                "r": lora_config.get("r", 8),
                "dropout": lora_config.get("dropout", 0),
                "trainable_modules": lora_config.get("trainable_modules", [])
            }

        payload = {
            "name": name,
            "dataset": {
                "id": dataset_id,
                "prompt_column": prompt_column,
                "answer_column": answer_column
            },
            "base_model_id": model,
            "hyperparameters": hyperparameters
        }
        print(f"Payload: {payload}")
        response = requests.post(
            f"{self.base_url}/jobs",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload
        )

        if response.status_code != 200:
            print(f"\n✗ Error creating fine-tuning job:")
            print(f"Status Code: {response.status_code}")
            print(f"URL: {self.base_url}/jobs")
            print(f"Response: {response.text}")
            print(f"Payload sent: {payload}")

            # Special handling for base_model_id errors
            if "base_model_id" in response.text:
                print("\nNote: The base_model_id should be a UUID, not a model name string.")
                print("Please check the Nscale console at https://console.nscale.com/models")
                print("to find the correct base model ID for your desired model.")

            # Special handling for 500 errors
            elif response.status_code == 500:
                print("\n" + "=" * 70)
                print("⚠️  500 INTERNAL SERVER ERROR")
                print("=" * 70)
                print("This appears to be an API-side issue with Nscale.")
                print("\nThe request payload is valid (dataset and model IDs exist),")
                print("but the server is returning an internal error.")
                print("\nNext steps:")
                print("  1. Check Nscale console: https://console.nscale.com")
                print("  2. Try again in a few minutes (may be temporary)")
                print("  3. Contact Nscale support if issue persists")
                print(f"  4. Reference: ISSUE_REPORT.md for detailed diagnostics")
                print("=" * 70)

        response.raise_for_status()
        result = response.json()

        # Debug: Print the full response to see what we got
        print(f"\nDEBUG - Full API Response:")
        print(f"  Response type: {type(result)}")
        print(f"  Response keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        print(f"  Full response: {result}")
        print()

        job_id = result.get('id')
        if not job_id:
            print("⚠ Warning: API response does not contain 'id' field")
            print(f"Response received: {result}")
            raise ValueError("API response missing 'id' field")

        print(f"✓ Fine-tuning job created. Job ID: {job_id}")
        return result

    def list_jobs(self, limit: int = 10, status: Optional[str] = None) -> List[Dict]:
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

        response = requests.get(
            f"{self.base_url}/jobs",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        result = response.json()
        return result.get("data", [])

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get the status of a fine-tuning job.

        Args:
            job_id: ID of the job

        Returns:
            Dictionary containing job status and metadata
        """
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_job_metrics(self, job_id: str) -> List[Dict]:
        """
        Get training metrics for a job.

        Args:
            job_id: ID of the job

        Returns:
            List of metric dictionaries
        """
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/metrics",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def monitor_job(self, job_id: str, poll_interval: int = 30) -> Dict:
        """
        Monitor a fine-tuning job until completion.

        Args:
            job_id: ID of the job to monitor
            poll_interval: Seconds between status checks

        Returns:
            Final job status
        """
        print(f"\nMonitoring job {job_id}...")
        print(f"You can also monitor progress at: https://console.nscale.com/jobs/{job_id}\n")

        with tqdm(desc="Training", unit="check") as pbar:
            while True:
                status = self.get_job_status(job_id)
                state = status.get("status", "unknown")

                pbar.set_postfix_str(f"Status: {state}")

                if state in ["succeeded", "completed"]:
                    print("\n✓ Training completed successfully!")
                    self._print_metrics(job_id)
                    return status
                elif state in ["failed", "cancelled"]:
                    print(f"\n✗ Training {state}")
                    if "error" in status:
                        print(f"Error: {status['error']}")
                    return status

                time.sleep(poll_interval)
                pbar.update(1)

    def _print_metrics(self, job_id: str) -> None:
        """Print training metrics for a job."""
        try:
            metrics = self.get_job_metrics(job_id)
            if metrics:
                print("\nTraining Metrics:")
                latest = metrics[-1]
                print(f"  Train Loss: {latest.get('train_loss', 'N/A')}")
                print(f"  Eval Loss: {latest.get('eval_loss', 'N/A')}")
                print(f"  Perplexity: {latest.get('perplexity', 'N/A')}")
        except Exception as e:
            print(f"Could not fetch metrics: {e}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using Nscale")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Path to training CSV file (overrides config)"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="Path to validation CSV file (overrides config)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job until completion"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Monitor existing job by ID"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available base models for fine-tuning"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_token = os.getenv("NSCALE_TOKEN") or os.getenv("NSCALE_API_TOKEN")
    organization_id = os.getenv("ORGANIZATION_ID")

    if not api_token:
        raise ValueError(
            "NSCALE_TOKEN not found. Please set it in .env file or environment."
        )

    if not organization_id:
        raise ValueError(
            "ORGANIZATION_ID not found. Please set it in .env file or environment.\n"
            "You can find your organization ID in the Nscale console."
        )

    # Load configuration
    config = load_config(args.config)

    # Initialize client
    client = NscaleClient(
        api_token=api_token,
        organization_id=organization_id,
        base_url=config.get("api", {}).get("fine_tuning_base_url", "https://fine-tuning.api.nscale.com/api/v1")
    )

    # If listing models
    if args.list_models:
        print("Available base models for fine-tuning:")
        print("=" * 60)
        try:
            models = client.list_models()
            for model in models:
                print(f"\nModel ID: {model.get('id')}")
                print(f"Name: {model.get('name', 'N/A')}")
                print(f"Description: {model.get('description', 'N/A')}")
        except Exception as e:
            print(f"Error fetching models: {e}")
        return

    # If monitoring existing job
    if args.job_id:
        print(f"Monitoring existing job: {args.job_id}")
        client.monitor_job(args.job_id)
        return

    # Determine file paths
    train_file = args.train_file or config["dataset"]["train_file"]
    val_file = args.val_file or config["dataset"]["validation_file"]

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not Path(val_file).exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    # Step 1: Create dataset (with file uploads)
    print("=" * 60)
    print("Step 1: Creating dataset")
    print("=" * 60)
    dataset_resp = client.create_dataset(
        train_file_path=train_file,
        validation_file_path=val_file,
        name=config["dataset"].get("name"),
        prompt_column=config["dataset"]["prompt_column"],
        answer_column=config["dataset"]["answer_column"]
    )

    # Step 2: Create fine-tuning job
    print("\n" + "=" * 60)
    print("Step 2: Creating fine-tuning job")
    print("=" * 60)

    hyperparameters = {
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "n_epochs": config["training"]["num_epochs"],
        "n_evals": config["training"]["num_evaluations"],  # API uses n_evals not n_evaluations
        "weight_decay": config["training"]["weight_decay"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "mask_prompt_labels": config["training"].get("mask_prompt_labels", False),
        "best_checkpoints": config["training"].get("best_checkpoints", True)
    }

    job_resp = client.create_finetune_job(
        dataset_id=dataset_resp["id"],
        model=config["model"]["base_model"],
        hyperparameters=hyperparameters,
        name=config.get("job", {}).get("name"),
        prompt_column=config["dataset"]["prompt_column"],
        answer_column=config["dataset"]["answer_column"],
        lora_config=config.get("lora")
    )

    job_id = job_resp["id"]
    print(f"\n{'=' * 60}")
    print(f"Fine-tuning job started!")
    print(f"{'=' * 60}")
    print(f"Job ID: {job_id}")
    print(f"Console: https://console.nscale.com/jobs/{job_id}")

    # Save job ID to file for later reference
    with open("last_job_id.txt", "w") as f:
        f.write(job_id)
    print(f"Job ID saved to: last_job_id.txt")

    # Step 3: Monitor job (if requested)
    if args.monitor:
        print(f"\n{'=' * 60}")
        print("Step 3: Monitoring job")
        print(f"{'=' * 60}")
        final_status = client.monitor_job(job_id)

        if final_status.get("status") == "succeeded":
            fine_tuned_model = final_status.get("fine_tuned_model")
            if fine_tuned_model:
                print(f"\n✓ Fine-tuned model: {fine_tuned_model}")
                print(f"You can now use this model for inference!")

                # Save model ID
                with open("fine_tuned_model.txt", "w") as f:
                    f.write(fine_tuned_model)
                print(f"Model ID saved to: fine_tuned_model.txt")
    else:
        print("\nTo monitor this job later, run:")
        print(f"  python finetune.py --job-id {job_id} --monitor")


if __name__ == "__main__":
    main()
