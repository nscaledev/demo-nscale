#!/usr/bin/env python3
"""
Fine-tuning script for Nscale API.

This script handles the complete fine-tuning workflow:
1. Create a dataset (uploads training and validation CSV files)
2. Launch a fine-tuning job
3. Monitor training progress
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import yaml

from nscale import FineTuningClient
from nscale.exceptions import ServerError, NscaleError


def load_config(config_path: str = "config.yaml") -> dict:
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
    client = FineTuningClient(
        api_token=api_token,
        organization_id=organization_id,
    )

    try:
        # If listing models
        if args.list_models:
            print("Available base models for fine-tuning:")
            print("=" * 60)
            models = client.jobs.list_base_models()
            for model in models:
                print(f"\nModel ID: {model.get('id')}")
                print(f"Name: {model.get('name', 'N/A')}")
                print(f"Description: {model.get('description', 'N/A')}")
            return

        # If monitoring existing job
        if args.job_id:
            print(f"Monitoring existing job: {args.job_id}")
            client.jobs.monitor_job(args.job_id)
            return

        # Determine file paths
        train_file = args.train_file or config["dataset"]["train_file"]
        val_file = args.val_file or config["dataset"]["validation_file"]

        if not Path(train_file).exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not Path(val_file).exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")

        # Step 1: Create dataset
        print("=" * 60)
        print("Step 1: Creating dataset")
        print("=" * 60)

        dataset = client.datasets.create(
            train_file_path=train_file,
            validation_file_path=val_file,
            name=config["dataset"].get("name"),
        )

        # Step 2: Create fine-tuning job
        print("\n" + "=" * 60)
        print("Step 2: Creating fine-tuning job")
        print("=" * 60)

        # Build hyperparameters from config
        hyperparameters = {
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "n_epochs": config["training"]["num_epochs"],
            "n_evals": config["training"]["num_evaluations"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "mask_prompt_labels": config["training"].get("mask_prompt_labels", False),
            "best_checkpoints": config["training"].get("best_checkpoints", True),
        }

        # Add LoRA config if enabled
        lora_config = config.get("lora", {})
        if lora_config and lora_config.get("enabled"):
            hyperparameters["lora"] = {
                "enabled": True,
                "alpha": lora_config.get("alpha", 16),
                "r": lora_config.get("r", 8),
                "dropout": lora_config.get("dropout", 0),
                "trainable_modules": lora_config.get("trainable_modules", []),
            }

        try:
            job = client.jobs.create_job(
                dataset_id=dataset["id"],
                base_model_id=config["model"]["base_model"],
                hyperparameters=hyperparameters,
                prompt_column=config["dataset"]["prompt_column"],
                answer_column=config["dataset"]["answer_column"],
                name=config.get("job", {}).get("name"),
            )
        except ServerError as e:
            # Job creation failed with 500 error (known API issue)
            print("\n⚠️  Job creation failed due to API-side issue")
            print("See ISSUE_REPORT.md and TESTING_SUMMARY.md for details")
            return

        job_id = job["id"]
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
            final_status = client.jobs.monitor_job(job_id)

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

    except NscaleError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
