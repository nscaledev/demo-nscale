#!/usr/bin/env python3
"""
Display detailed information about a fine-tuning job.
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

from nscale import FineTuningClient
from nscale.exceptions import NscaleError


def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return timestamp_str


def print_job_details(job_id: str, api_token: str, org_id: str):
    """Print detailed job information."""
    try:
        client = FineTuningClient(
            api_token=api_token,
            organization_id=org_id
        )

        # Fetch job details
        job = client.jobs.get_job(job_id)

        # Print header
        print()
        print("=" * 80)
        print(f"Fine-Tuning Job Details: {job_id}")
        print("=" * 80)
        print()

        # Basic info
        print("📋 Basic Information")
        print("─" * 80)
        print(f"  Name:             {job.get('name', 'N/A')}")
        print(f"  Status:           {job.get('status', 'N/A')}")
        print(f"  Job ID:           {job_id}")
        print(f"  Base Model ID:    {job.get('base_model_id', 'N/A')}")
        print(f"  Fine-tuned Model: {job.get('fine_tuned_model', 'N/A')}")
        print()

        # Timestamps
        print("⏰ Timeline")
        print("─" * 80)
        if 'created_at' in job:
            print(f"  Created:          {format_timestamp(job['created_at'])}")
        if 'started_at' in job:
            print(f"  Started:          {format_timestamp(job['started_at'])}")
        if 'finished_at' in job:
            print(f"  Finished:         {format_timestamp(job['finished_at'])}")
        print()

        # Dataset info
        dataset_info = job.get('dataset', {})
        if dataset_info:
            print("📊 Dataset")
            print("─" * 80)
            print(f"  Dataset ID:       {dataset_info.get('id', 'N/A')}")
            print(f"  Prompt Column:    {dataset_info.get('prompt_column', 'N/A')}")
            print(f"  Answer Column:    {dataset_info.get('answer_column', 'N/A')}")
            print()

        # Hyperparameters
        hyperparams = job.get('hyperparameters', {})
        if hyperparams:
            print("⚙️  Hyperparameters")
            print("─" * 80)
            print(f"  Batch Size:       {hyperparams.get('batch_size', 'N/A')}")
            print(f"  Learning Rate:    {hyperparams.get('learning_rate', 'N/A')}")
            print(f"  Epochs:           {hyperparams.get('n_epochs', 'N/A')}")
            print(f"  Weight Decay:     {hyperparams.get('weight_decay', 'N/A')}")

            lora = hyperparams.get('lora', {})
            if lora:
                print(f"  LoRA Enabled:     {lora.get('enabled', False)}")
                if lora.get('enabled'):
                    print(f"  LoRA Alpha:       {lora.get('alpha', 'N/A')}")
                    print(f"  LoRA R:           {lora.get('r', 'N/A')}")
                    print(f"  LoRA Dropout:     {lora.get('dropout', 'N/A')}")
            print()

        # Progress
        if 'progress' in job:
            print("📈 Progress")
            print("─" * 80)
            progress = job['progress']
            if isinstance(progress, (int, float)):
                print(f"  Completion:       {progress * 100:.1f}%")
            else:
                print(f"  Completion:       {progress}")
            print()

        # Metrics
        print("📊 Training Metrics")
        print("─" * 80)
        console_url = f"https://console.nscale.com/organization/{org_id}/fine-tuning/jobs/{job_id}/overview"
        try:
            metrics = client.jobs.get_metrics(job_id)
            if metrics and isinstance(metrics, list) and len(metrics) > 0:
                latest = metrics[-1]
                print(f"  Latest Step:      {latest.get('step', 'N/A')}")
                print(f"  Train Loss:       {latest.get('train_loss', 'N/A')}")
                print(f"  Eval Loss:        {latest.get('eval_loss', 'N/A')}")
                print(f"  Perplexity:       {latest.get('perplexity', 'N/A')}")
                print(f"  Total Steps:      {len(metrics)}")
            else:
                print("  No metrics available yet")
                print(f"  View in console:  {console_url}")
        except Exception as e:
            print(f"  Could not fetch metrics: {e}")
            print(f"  View in console:  {console_url}")
        print()

        # Error info
        if 'error' in job and job['error']:
            print("❌ Error Information")
            print("─" * 80)
            print(f"  {job['error']}")
            print()

        # Links
        print("🔗 Links")
        print("─" * 80)
        print(f"  Console:          {console_url}")
        print()

        print("=" * 80)
        print()

        client.close()

    except NscaleError as e:
        print(f"✗ Error fetching job details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Display fine-tuning job details")
    parser.add_argument("--job-id", help="Job ID to query (uses last_job_id.txt if not provided)")
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    api_token = os.getenv("NSCALE_TOKEN")
    org_id = os.getenv("ORGANIZATION_ID")

    if not api_token:
        print("✗ Error: NSCALE_TOKEN not set in .env")
        sys.exit(1)

    if not org_id:
        print("✗ Error: ORGANIZATION_ID not set in .env")
        sys.exit(1)

    # Get job ID from argument or last_job_id.txt
    job_id = args.job_id
    if not job_id:
        if os.path.exists("last_job_id.txt"):
            with open("last_job_id.txt", "r") as f:
                job_id = f.read().strip()
            print(f"Using job ID from last_job_id.txt: {job_id}")
        else:
            print("✗ Error: No job ID provided and last_job_id.txt not found")
            print()
            print("Usage:")
            print("  python job_info.py --job-id <job-id>")
            print("  OR")
            print("  Run a job first with 'make finetune' to create last_job_id.txt")
            sys.exit(1)

    print_job_details(job_id, api_token, org_id)


if __name__ == "__main__":
    main()
