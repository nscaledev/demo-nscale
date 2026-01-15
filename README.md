# Nscale Fine-Tuning and Inference Demo

Professional Python client library and demo scripts for Nscale's fine-tuning and inference APIs.

## Features

- **Python Client Library** - Production-ready `nscale` package with clean architecture
- **Fine-Tuning** - Create datasets and train models with LoRA
- **Inference** - Run predictions with deployed models (OpenAI-compatible)
- **Monitoring** - Track training progress in real-time

## Quick Start

```bash
# 1. Setup environment
make setup install

# 2. Add your NSCALE_TOKEN to .env, then auto-fetch org ID
make get-org-id

# 3. Create sample data and fine-tune
make data finetune-monitor

# 4. Run inference
MODEL="Qwen/Qwen3-4B-Instruct-2507" make inference
```

See [`make help`](Makefile) for all commands.

## Project Structure

```
demo-nscale/
├── nscale/                   # Python client library
│   ├── base.py               # Base service class
│   ├── platform.py           # Core client with auth & retries
│   ├── files.py              # File uploads
│   ├── datasets.py           # Dataset management
│   ├── finetuning.py         # Job management
│   ├── inference.py          # Inference client
│   └── exceptions.py         # Custom exceptions
├── tests/                    # Test suite (60 tests)
│   ├── test_platform.py      # Platform client tests
│   ├── test_files.py         # File service tests
│   ├── test_datasets.py      # Dataset service tests
│   ├── test_finetuning.py    # Fine-tuning service tests
│   ├── test_inference.py     # Inference client tests
│   └── test_get_org_id.py    # Organization ID fetching tests
├── finetune.py               # Fine-tuning script
├── inference.py              # Inference script
├── prepare_dataset.py        # Dataset preparation
├── get_org_id.py             # Organization ID auto-fetch
├── config.yaml               # Training configuration
└── data/                     # Training datasets
```

## Using the Nscale Client Library

### Fine-Tuning

```python
from nscale import FineTuningClient

with FineTuningClient(api_token=token, organization_id=org_id) as client:
    # Create dataset
    dataset = client.datasets.create(
        train_file_path="train.csv",
        validation_file_path="val.csv"
    )

    # Create training job
    job = client.jobs.create_job(
        dataset_id=dataset["id"],
        base_model_id="5b912da4-4a68-43eb-9224-38239535d934",
        hyperparameters={
            "batch_size": 32,
            "learning_rate": 0.00001,
            "n_epochs": 3,
            "lora": {"enabled": True, "alpha": 16, "r": 8}
        }
    )

    # Monitor progress
    final_status = client.jobs.monitor_job(job["id"])
```

### Inference

```python
from nscale import InferenceClient

with InferenceClient(api_token=token) as client:
    # List available models
    models = client.list_models()

    # Generate answer
    answer = client.generate_answer(
        model="Qwen/Qwen3-4B-Instruct-2507",
        question="What is machine learning?"
    )
    print(answer)
```

## Configuration

### API Credentials (`.env`)

```bash
NSCALE_TOKEN=your_api_token
ORGANIZATION_ID=auto-fetched-uuid
```

**Getting Your Credentials:**

1. **API Token**: Get from [console.nscale.com/settings](https://console.nscale.com/settings)
2. **Organization ID**: Automatically fetched by running `make get-org-id`

The `get-org-id` command calls the identity API (`https://identity.nks.europe-west4.nscale.com/api/v1/acl`) to fetch all organizations associated with your token and automatically selects the first one.

### Training Config (`config.yaml`)

```yaml
model:
  base_model: "5b912da4-4a68-43eb-9224-38239535d934"  # Meta-Llama-3.1-8B-Instruct

training:
  batch_size: 32
  learning_rate: 0.00001
  num_epochs: 3
  num_evaluations: 6
  weight_decay: 0.03

lora:
  enabled: true
  alpha: 16
  r: 8
  dropout: 0
```

## Command Line Usage

### Fine-Tuning

```bash
# Start fine-tuning with monitoring
python finetune.py --monitor

# List available base models
python finetune.py --list-models

# Monitor existing job
python finetune.py --job-id <job-id> --monitor

# Show detailed job information
python job_info.py                    # Uses last_job_id.txt
python job_info.py --job-id <job-id>  # Specific job
```

### Inference

```bash
# List inference endpoints
python inference.py --list-models

# Interactive Q&A
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --interactive

# Single question
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --question "What is AI?"
```

### Dataset Preparation

```bash
# Create sample datasets
python prepare_dataset.py --create-sample

# Convert your data
python prepare_dataset.py --input data.json --output train.csv
```

## Make Commands

Nscale Demo - Available Commands
================================= 
     Setup:
       make setup          - Create virtual environment                                               
       make install        - Install dependencies with uv                                             
       make get-org-id     - Get your Nscale organization ID                                          
       make data           - Create sample training/validation data                                   
                                                                                                      
     Testing:                                                                                         
       make test-suite     - Run pytest test suite                                                    
       make test-inference - Test inference with a single question                                    
                                                                                                      
     Fine-tuning:
       make finetune       - Start fine-tuning job
       make finetune-monitor - Start fine-tuning with monitoring
       make monitor        - Monitor existing job (uses last_job_id.txt)
       make status         - Check status of last job
       make job-info       - Show detailed job info (uses last_job_id.txt)                                                 
                                                                                                      
     Inference:                                                                                       
       make list-models    - List available inference endpoints                                       
       make inference      - Run interactive inference (requires MODEL env var)                       
       make interactive    - Same as 'make inference'                                                 
                                                                                                      
     Cleanup:                                                                                         
       make clean          - Remove generated files and cache                                         
       make clean-all      - Remove everything including venv   

## Architecture

The `nscale` package provides:

- **BaseService** - Common base class for all service clients
- **Platform Client** - Authentication, retries, error handling
- **Service Clients** - Files, datasets, jobs (all inherit from BaseService)
- **Inference Client** - Standalone client for inference operations
- **Custom Exceptions** - Type-safe error handling
- **Context Managers** - Automatic resource cleanup

### Service Architecture

```python
BaseService
    ├── FileService           # File upload/management
    ├── DatasetService        # Dataset creation (uses FileService)
    └── FineTuningService     # Training job management

NscaleClient                  # Core HTTP client (composition)
InferenceClient               # Standalone inference client
```

All services use **composition** rather than inheritance - they accept a `NscaleClient` instance and delegate HTTP operations to it. This provides better separation of concerns and testability.

## Examples

### End-to-End Workflow

```python
from nscale import FineTuningClient, InferenceClient

# Fine-tune a model
ft_client = FineTuningClient(api_token=token, organization_id=org_id)

dataset = ft_client.datasets.create("train.csv", "val.csv")
job = ft_client.jobs.create_job(
    dataset["id"],
    "5b912da4-4a68-43eb-9224-38239535d934",
    {"batch_size": 32, "n_epochs": 3}
)
ft_client.jobs.monitor_job(job["id"])
ft_client.close()

# Run inference (with deployed models, not fine-tuned ones yet)
inf_client = InferenceClient(api_token=token)
answer = inf_client.generate_answer(
    model="Qwen/Qwen3-4B-Instruct-2507",
    question="Explain the model we just trained"
)
print(answer)
inf_client.close()
```

### Error Handling

```python
from nscale import FineTuningClient
from nscale.exceptions import ServerError, ValidationError

try:
    client = FineTuningClient(api_token=token, organization_id=org_id)
    job = client.jobs.create_job(...)
except ValidationError as e:
    print(f"Invalid parameters: {e}")
except ServerError as e:
    print(f"Server error: {e}")
    print(f"Status code: {e.status_code}")
```

## Troubleshooting

**Missing Organization ID**
```bash
# Ensure NSCALE_TOKEN is set in .env first
make get-org-id
```

If the command fails:
- Verify your `NSCALE_TOKEN` is correct
- Check you have access to the identity API
- The token must have organization access permissions

**Module Import Errors**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**File Not Found**
```bash
make data  # Create sample datasets
```

**Job Creation Issues**
- Verify organization ID is correct UUID format (auto-fetched by `make get-org-id`)
- Check API token has fine-tuning permissions
- Ensure datasets are created successfully before job creation

**Identity API Issues**
```bash
# Test the identity API directly
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://identity.nks.europe-west4.nscale.com/api/v1/acl
```

## Resources

- [Setup Guide](SETUP.md) - Quick setup instructions
- [Nscale Documentation](https://docs.nscale.com/)
- [Nscale Console](https://console.nscale.com/)
- [Fine-Tuning Cookbook](https://docs.nscale.com/docs/cookbooks/pubmedqa-fine-tuning)

## License

This is a demo project for educational purposes.

## Support

- **Demo Issues**: Open an issue in this repository
- **Nscale API Issues**: Contact Nscale support
- **API Access**: Check https://console.nscale.com/settings
