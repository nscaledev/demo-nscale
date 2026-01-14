# Nscale Fine-Tuning and Inference Demo

A comprehensive demo project for fine-tuning large language models and running inference using the Nscale API.

## Features

- **Dataset Preparation**: Convert your data to Nscale-compatible CSV format
- **Fine-Tuning**: Train models with customizable hyperparameters and LoRA configuration
- **Inference**: Run queries with deployed models through Nscale's inference endpoints
- **Monitoring**: Track training progress and metrics in real-time

## Quick Start

### Using Make (Recommended)

```bash
# 1. Setup environment
make setup
make install
make get-org-id  # Get and configure your organization ID
make data

# 2. Start fine-tuning with monitoring
make finetune-monitor

# 3. List available inference models
make list-models

# 4. Run interactive inference with a model
MODEL="Qwen/Qwen3-4B-Instruct-2507" make inference
```

See all available commands: `make help`

### Using Python Directly

See the detailed [Step-by-Step Guide](#step-by-step-guide) below.

## Project Structure

```
demo-nscale/
├── .env                      # API credentials
├── Makefile                  # Make commands for easy workflow
├── config.yaml               # Training configuration
├── requirements.txt          # Python dependencies
├── prepare_dataset.py        # Dataset preparation script
├── finetune.py              # Fine-tuning script
├── inference.py             # Inference script
├── data/
│   ├── train.csv            # Training dataset
│   ├── validation.csv       # Validation dataset
│   └── example_questions.json  # Example questions
└── venv/                    # Virtual environment
```

## Setup

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv pip install -r requirements.txt
```

Or using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials

Your `.env` file is already set up with your API token:

```bash
NSCALE_TOKEN=your_token_here
```

### 4. Customize Configuration

Edit `config.yaml` to adjust training parameters:

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"

training:
  batch_size: 32
  learning_rate: 0.00001
  num_epochs: 3
  num_evaluations: 6

lora:
  enabled: true
  alpha: 16
  r: 8
  dropout: 0
```

## Step-by-Step Guide

This guide walks you through the complete workflow from setup to inference.

### Phase 1: Environment Setup

#### Step 1: Create Virtual Environment

```bash
make setup
```

Or manually:

```bash
python3 -m venv venv
```

This creates an isolated Python environment in the `venv/` directory.

#### Step 2: Install Dependencies

```bash
make install
```

Or manually:

```bash
source venv/bin/activate  # Activate the virtual environment
uv pip install -r requirements.txt  # Install packages
```

This installs: `requests`, `pandas`, `python-dotenv`, `pyyaml`, `tqdm`

#### Step 3: Verify API Token

Check that your `.env` file contains your Nscale token:

```bash
cat .env
```

Should show:
```
NSCALE_TOKEN=eyJhbGciOiJ...
ORGANIZATION_ID=your_organization_id_here
```

#### Step 3.5: Get Your Organization ID

Fetch your organization ID automatically:

```bash
make get-org-id
```

Or manually:

```bash
source venv/bin/activate
python get_org_id.py
```

This will:
- Fetch all organizations associated with your token
- Display organization names and IDs
- Optionally update your `.env` file automatically

Alternatively, find your organization ID at: https://console.nscale.com/settings

### Phase 2: Data Preparation

#### Step 4: Create Sample Dataset

```bash
make data
```

Or manually:

```bash
source venv/bin/activate
python prepare_dataset.py --create-sample
```

This generates:
- `data/train.csv` - 50 training examples
- `data/validation.csv` - 2 validation examples

#### Step 5: Verify Data Format

Check the generated CSV files:

```bash
head -n 3 data/train.csv
```

Should show columns: `question,answer`

**For Your Own Data:**

If you have custom data, format it as JSON:

```json
[
  {
    "question": "Your question here?",
    "answer": "Your answer here."
  }
]
```

Then convert it:

```bash
python prepare_dataset.py --input your_data.json --output data/train.csv
```

### Phase 3: Fine-Tuning

#### Step 6: Review Configuration

Check `config.yaml` to ensure settings are appropriate:

```bash
cat config.yaml
```

Key settings:
- `base_model`: Which model to fine-tune (default: Llama-3.1-8B-Instruct)
- `num_epochs`: How many training epochs (default: 3)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.00001)

#### Step 7: Start Fine-Tuning

```bash
make finetune-monitor
```

Or manually:

```bash
source venv/bin/activate
python finetune.py --monitor
```

This will:
1. Upload `data/train.csv` to Nscale
2. Upload `data/validation.csv` to Nscale
3. Create a dataset configuration
4. Launch a fine-tuning job
5. Save job ID to `last_job_id.txt`
6. Monitor progress in real-time

**Expected Output:**

```
============================================================
Step 1: Uploading files
============================================================
Uploading data/train.csv...
✓ Uploaded successfully. File ID: file-abc123

Uploading data/validation.csv...
✓ Uploaded successfully. File ID: file-xyz789

============================================================
Step 2: Creating dataset
============================================================
Creating dataset configuration...
✓ Dataset created. Dataset ID: dataset-def456

============================================================
Step 3: Creating fine-tuning job
============================================================
Creating fine-tuning job for model: meta-llama/Llama-3.1-8B-Instruct...
✓ Fine-tuning job created. Job ID: job-ghi789

============================================================
Fine-tuning job started!
============================================================
Job ID: job-ghi789
Console: https://console.nscale.com/jobs/job-ghi789

============================================================
Step 4: Monitoring job
============================================================
Monitoring job job-ghi789...
You can also monitor progress at: https://console.nscale.com/jobs/job-ghi789

Training: 100%|████████| Status: succeeded
```

#### Step 8: Monitor Training (Alternative)

If you started training without monitoring, check status:

```bash
make status
```

Or monitor an existing job:

```bash
make monitor
```

### Phase 4: Inference

**Note**: Inference uses deployed models from Nscale's inference endpoints, not fine-tuned models (which are not yet supported for inference).

#### Step 9: List Available Models

First, see what models are available for inference:

```bash
make list-models
```

This will show all available inference endpoints with their pricing and context lengths.

#### Step 10: Run Interactive Inference

Start an interactive session with a model:

```bash
MODEL="Qwen/Qwen3-4B-Instruct-2507" make inference
```

Or manually:

```bash
source venv/bin/activate
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --interactive
```

**Example Session:**

```
============================================================
Interactive Mode - Model: Qwen/Qwen3-4B-Instruct-2507
============================================================
Type your questions below. Type 'quit' or 'exit' to stop.

Question: What is the primary function of mitochondria in cells?

Generating answer...

Answer: The primary function of mitochondria in cells is to produce energy
in the form of ATP through cellular respiration. They convert nutrients
into energy that cells use to power various biological processes.

------------------------------------------------------------

Question: exit
Goodbye!
```

#### Step 11: Test Single Question

```bash
MODEL="Qwen/Qwen3-4B-Instruct-2507" make test
```

Or manually:

```bash
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --question "What causes type 2 diabetes?"
```

### Phase 5: Iteration

#### Step 12: Improve Your Fine-Tuned Model

Based on your needs, you can improve the fine-tuning process:

1. **Add more training data**: Update `data/train.csv`
2. **Adjust hyperparameters**: Edit `config.yaml`
3. **Use more epochs**: Increase `num_epochs` in config
4. **Fine-tune again**: Run `make finetune-monitor`

#### Step 13: Try Different Inference Models

Test different models to see which works best for your use case:

```bash
# Try a different model
MODEL="Qwen/Qwen3-8B-Instruct-2507" python inference.py --question "Test question"

# Or use a larger model
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507" python inference.py --interactive
```

### Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│                    Complete Workflow                     │
└─────────────────────────────────────────────────────────┘

  Setup Phase
  ├── make setup       (Create virtual environment)
  ├── make install     (Install dependencies)
  └── make data        (Create sample datasets)

  Fine-Tuning Phase
  └── make finetune-monitor  (Train model, ~10-30 minutes)

  Inference Phase
  ├── make list-models  (List available models)
  ├── MODEL=<id> make inference  (Interactive Q&A)
  └── MODEL=<id> make test       (Single question)

  Iteration Phase
  ├── Edit data/train.csv or config.yaml
  └── make finetune-monitor (Retrain)
```

## Make Commands Reference

Run `make help` to see all available commands:

```bash
make help          # Show all commands
make setup         # Create virtual environment
make install       # Install dependencies
make data          # Create sample data
make finetune      # Start training (no monitoring)
make finetune-monitor  # Start training with monitoring
make monitor       # Monitor existing job
make status        # Check last job status
make list-models   # List available inference models
make inference     # Interactive mode (requires MODEL env var)
make interactive   # Same as inference
make test          # Test with single question (requires MODEL env var)
make clean         # Remove generated files
make clean-all     # Remove everything including venv
```

## Usage

### Step 1: Prepare Your Dataset

#### Option A: Use Sample Data

Generate sample training and validation datasets:

```bash
python prepare_dataset.py --create-sample
```

This creates:
- `data/train.csv` - 50 sample training examples
- `data/validation.csv` - 2 sample validation examples

#### Option B: Convert Your Own Data

Prepare your data in JSON or JSONL format with `question` and `answer` fields:

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of AI that enables systems to learn from data..."
  }
]
```

Then convert to CSV:

```bash
python prepare_dataset.py --input your_data.json --output data/train.csv
```

### Step 2: Fine-Tune a Model

Launch a fine-tuning job:

```bash
python finetune.py
```

To monitor training progress in real-time:

```bash
python finetune.py --monitor
```

The script will:
1. Upload your training and validation files
2. Create a dataset configuration
3. Launch a fine-tuning job
4. Save the job ID to `last_job_id.txt`
5. (Optional) Monitor progress until completion

#### Monitor an Existing Job

```bash
python finetune.py --job-id <job_id> --monitor
```

Or load the last job ID:

```bash
python finetune.py --job-id $(cat last_job_id.txt) --monitor
```

### Step 3: Run Inference

#### List Available Models

First, list available inference endpoints:

```bash
python inference.py --list-models
```

This will show all available models with pricing and context lengths.

#### Interactive Mode

Start an interactive question-answering session with a model:

```bash
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --interactive
```

Example session:

```
Interactive Mode - Model: Qwen/Qwen3-4B-Instruct-2507
Type your questions below. Type 'quit' or 'exit' to stop.

Question: What is the primary function of mitochondria?

Answer: The primary function of mitochondria is to produce energy in the
form of ATP through cellular respiration...
```

#### Single Question

Ask a single question:

```bash
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --question "What causes type 2 diabetes?"
```

#### Try Different Models

You can use any available model from the list:

```bash
# Use a larger model for more complex questions
python inference.py --model "Qwen/Qwen3-8B-Instruct-2507" --interactive

# Or a smaller, faster model
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --interactive
```

## Configuration Reference

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.00001 | Learning rate for optimization |
| `num_epochs` | 3 | Number of training epochs |
| `num_evaluations` | 6 | Number of evaluation steps |
| `weight_decay` | 0.03 | Weight decay for regularization |
| `warmup_ratio` | 0 | Warmup ratio for learning rate |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable LoRA fine-tuning |
| `alpha` | 16 | LoRA alpha parameter |
| `r` | 8 | LoRA rank |
| `dropout` | 0 | LoRA dropout rate |

## Examples

### Complete Workflow

```bash
# 1. Prepare dataset
python prepare_dataset.py --create-sample

# 2. Launch fine-tuning with monitoring
python finetune.py --monitor

# 3. List available inference models
python inference.py --list-models

# 4. Run interactive inference with a model
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --interactive

# 5. Test with a single question
python inference.py --model "Qwen/Qwen3-4B-Instruct-2507" --question "What is AI?"
```

### Custom Data Pipeline

```bash
# Convert custom data
python prepare_dataset.py \
  --input my_data.jsonl \
  --output data/train.csv \
  --max-examples 2000

# Fine-tune with custom config
python finetune.py \
  --config my_config.yaml \
  --monitor

# Test with inference endpoint
python inference.py \
  --model "Qwen/Qwen3-8B-Instruct-2507" \
  --question "Test question"
```

## File Outputs

- `last_job_id.txt` - Most recent fine-tuning job ID
- `inference_results.json` - Batch inference results (when using batch mode)

## Monitoring

### Via Script

```bash
python finetune.py --job-id $(cat last_job_id.txt) --monitor
```

### Via Web Console

```
https://console.nscale.com/jobs/<job_id>
```

## Troubleshooting

### ModuleNotFoundError

Make sure you've activated the virtual environment and installed dependencies:

```bash
source venv/bin/activate
uv pip install -r requirements.txt
```

### API Token Issues

Verify your `.env` file contains:

```bash
NSCALE_TOKEN=your_actual_token_here
```

### File Not Found

Ensure you've created datasets:

```bash
python prepare_dataset.py --create-sample
```

### Job Monitoring

Check job status manually:

```bash
python finetune.py --job-id <job_id> --monitor
```

## Advanced Usage

### Custom Dataset Format

Modify the `example_to_qa_row()` function in `prepare_dataset.py` to handle your specific data format:

```python
def example_to_qa_row(ex: Dict) -> Optional[Dict[str, str]]:
    # Customize based on your data structure
    question = ex.get("input_text")
    answer = ex.get("output_text")

    if not question or not answer:
        return None

    return {"question": question, "answer": answer}
```

### Inference Parameters

Customize generation parameters in `inference.py`:

```python
response = client.chat_completion(
    model=model,
    messages=messages,
    max_tokens=1024,      # Increase for longer responses
    temperature=0.3,      # Lower for more deterministic outputs
    top_p=0.95           # Adjust nucleus sampling
)
```

## Resources

- [Nscale Documentation](https://docs.nscale.com/)
- [Nscale API Reference](https://docs.nscale.com/api)
- [Fine-Tuning Cookbook](https://docs.nscale.com/docs/cookbooks/pubmedqa-fine-tuning)
- [Nscale Console](https://console.nscale.com/)

## License

This is a demo project for educational purposes.

## Support

For issues with:
- **This demo**: Open an issue in this repository
- **Nscale API**: Contact Nscale support
- **API Access**: Check your Nscale account settings
