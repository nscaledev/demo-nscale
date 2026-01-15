.PHONY: help setup install get-org-id clean data finetune finetune-monitor monitor inference interactive batch test test-suite test-inference status job-info

# Default target
help:
	@echo "Nscale Demo - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create virtual environment"
	@echo "  make install        - Install dependencies with uv"
	@echo "  make get-org-id     - Get your Nscale organization ID"
	@echo "  make data           - Create sample training/validation data"
	@echo ""
	@echo "Testing:"
	@echo "  make test-suite     - Run pytest test suite"
	@echo "  make test-inference - Test inference with a single question"
	@echo ""
	@echo "Fine-tuning:"
	@echo "  make finetune       - Start fine-tuning job"
	@echo "  make finetune-monitor - Start fine-tuning with monitoring"
	@echo "  make monitor        - Monitor existing job (uses last_job_id.txt)"
	@echo "  make status         - Check status of last job"
	@echo "  make job-info       - Show detailed job info (uses last_job_id.txt)"
	@echo ""
	@echo "Inference:"
	@echo "  make list-models    - List available inference endpoints"
	@echo "  make inference      - Run interactive inference (requires MODEL env var)"
	@echo "  make interactive    - Same as 'make inference'"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files and cache"
	@echo "  make clean-all      - Remove everything including venv"
	@echo ""

# Setup commands
setup:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "✓ Virtual environment created at venv/"
	@echo ""
	@echo "Next step: Run 'make install'"

install:
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		. venv/bin/activate && uv pip install -r requirements.txt; \
	else \
		echo "uv not found, using pip..."; \
		. venv/bin/activate && pip install -r requirements.txt; \
	fi
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "Next step: Run 'make get-org-id' to configure your organization"

get-org-id:
	@echo "Fetching your organization ID..."
	@. venv/bin/activate && python get_org_id.py
	@echo ""
	@echo "Next step: Run 'make data' to create sample datasets"

data:
	@echo "Creating sample datasets..."
	@. venv/bin/activate && python prepare_dataset.py --create-sample
	@echo ""
	@echo "✓ Sample data created:"
	@echo "  - data/train.csv (50 examples)"
	@echo "  - data/validation.csv (2 examples)"
	@echo ""
	@echo "Next step: Run 'make finetune-monitor'"

# Fine-tuning commands
finetune:
	@echo "Starting fine-tuning job..."
	@. venv/bin/activate && python finetune.py
	@echo ""
	@echo "To monitor progress: make monitor"

finetune-monitor:
	@echo "Starting fine-tuning job with monitoring..."
	@. venv/bin/activate && python finetune.py --monitor

monitor:
	@if [ -f last_job_id.txt ]; then \
		JOB_ID=$$(cat last_job_id.txt); \
		echo "Monitoring job: $$JOB_ID"; \
		. venv/bin/activate && python finetune.py --job-id $$JOB_ID --monitor; \
	else \
		echo "Error: last_job_id.txt not found"; \
		echo "Run 'make finetune' first or specify job ID manually:"; \
		echo "  python finetune.py --job-id YOUR_JOB_ID --monitor"; \
	fi

status:
	@if [ -f last_job_id.txt ]; then \
		JOB_ID=$$(cat last_job_id.txt); \
		echo "Job ID: $$JOB_ID"; \
		echo "Console: https://console.nscale.com/jobs/$$JOB_ID"; \
	else \
		echo "No job ID found. Run 'make finetune' first."; \
	fi

job-info:
	@. venv/bin/activate && python job_info.py

# Inference commands
list-models:
	@echo "Fetching available inference endpoints..."
	@. venv/bin/activate && python inference.py --list-models

inference:
	@echo "Starting interactive inference..."
	@if [ -z "$$MODEL" ]; then \
		echo "Error: MODEL environment variable not set."; \
		echo ""; \
		echo "Usage:"; \
		echo "  1. List available models: make list-models"; \
		echo "  2. Run inference: MODEL=<model_id> make inference"; \
		echo ""; \
		echo "Example:"; \
		echo "  MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct make inference"; \
		exit 1; \
	else \
		echo "Using model: $$MODEL"; \
		. venv/bin/activate && python inference.py --model "$$MODEL" --interactive; \
	fi

interactive: inference

# Testing commands
test-suite:
	@echo "Running pytest test suite..."
	@. venv/bin/activate && pytest tests/ -v

test-inference:
	@echo "Testing inference with a sample question..."
	@if [ -z "$$MODEL" ]; then \
		echo "Error: MODEL environment variable not set."; \
		echo ""; \
		echo "Usage: MODEL=<model_id> make test-inference"; \
		echo "Example: MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct make test-inference"; \
		exit 1; \
	else \
		. venv/bin/activate && python inference.py \
			--model "$$MODEL" \
			--question "What is the primary function of mitochondria in cells?"; \
	fi

# Cleanup commands
clean:
	@echo "Cleaning generated files..."
	@rm -f last_job_id.txt fine_tuned_model.txt inference_results.json
	@rm -rf __pycache__ *.pyc
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned generated files"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf venv
	@echo "✓ Removed everything"
	@echo ""
	@echo "To start fresh, run: make setup"
