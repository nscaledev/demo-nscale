#!/usr/bin/env python3
"""
Inference script for Nscale models.

This script allows you to:
1. List available inference endpoints
2. Run inference with deployed models
3. Test models interactively
4. Batch process questions from a file
"""

import os
import requests
import argparse
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv


class NscaleInference:
    """Client for running inference with Nscale models."""

    def __init__(
        self,
        api_token: str,
        organization_id: str,
        base_url: str = "https://inference.api.nscale.com/v1"
    ):
        self.api_token = api_token
        self.organization_id = organization_id
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def list_models(self) -> List[Dict]:
        """
        List available inference models (OpenAI-compatible endpoint).

        Returns:
            List of available models with their metadata
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers
        )
        response.raise_for_status()
        result = response.json()
        return result.get("data", [])

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate text using a fine-tuned model.

        Args:
            model: Model ID (e.g., fine-tuned model ID)
            prompt: Input prompt/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            stop: List of stop sequences

        Returns:
            Dictionary containing generated text and metadata
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        if stop:
            payload["stop"] = stop

        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        Generate chat completion using a fine-tuned model.

        Args:
            model: Model ID
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary containing generated response
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        return response.json()


def interactive_mode(client: NscaleInference, model: str):
    """Run interactive question-answering session."""
    print(f"\n{'=' * 60}")
    print(f"Interactive Mode - Model: {model}")
    print(f"{'=' * 60}")
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nGenerating answer...")

            # Use chat completion format
            messages = [
                {"role": "user", "content": question}
            ]

            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )

            # Extract answer from response
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "No response")

            print(f"\nAnswer: {answer}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def batch_inference(
    client: NscaleInference,
    model: str,
    input_file: str,
    output_file: str
):
    """
    Run batch inference on questions from a file.

    Args:
        client: NscaleInference client
        model: Model ID
        input_file: Path to input file (JSON or text)
        output_file: Path to output file
    """
    print(f"Loading questions from {input_file}...")

    # Load questions
    questions = []
    with open(input_file, 'r') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                questions = [item.get("question", item) if isinstance(item, dict) else item for item in data]
            else:
                questions = [data.get("question", "")]
        else:
            questions = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(questions)} questions...")

    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing: {question[:50]}...")

        try:
            messages = [{"role": "user", "content": question}]
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )

            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            results.append({
                "question": question,
                "answer": answer,
                "model": model
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "model": model
            })

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


def single_inference(client: NscaleInference, model: str, question: str):
    """Run inference on a single question."""
    print(f"\nQuestion: {question}\n")
    print("Generating answer...")

    messages = [{"role": "user", "content": question}]

    response = client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )

    answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    print(f"\nAnswer: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with Nscale models")
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use for inference"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available inference endpoints/models"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to file with questions for batch processing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.json",
        help="Output file for batch results"
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
            "ORGANIZATION_ID not found. Please set it in .env file or environment."
        )

    # Initialize client
    client = NscaleInference(
        api_token=api_token,
        organization_id=organization_id
    )

    # List models if requested
    if args.list_models:
        print("Available inference models:")
        print("=" * 80)
        try:
            models = client.list_models()
            if not models:
                print("No models available.")
            else:
                for model in models:
                    print(f"\nModel ID: {model.get('id')}")
                    print(f"  Owner: {model.get('owned_by', 'N/A')}")
                    print(f"  Context Length: {model.get('context_length', 'N/A')}")
                    pricing = model.get('pricing', {})
                    if pricing:
                        print(f"  Pricing (per million):")
                        print(f"    Input: ${pricing.get('input', 'N/A')}")
                        print(f"    Output: ${pricing.get('output', 'N/A')}")
        except Exception as e:
            print(f"Error fetching models: {e}")
        return

    # Require model ID for inference operations
    model_id = args.model

    if not model_id:
        print("Error: No model ID provided.")
        print("\nTo get started:")
        print("  1. List available models: python inference.py --list-models")
        print("  2. Run inference: python inference.py --model <model_id> --question 'Your question'")
        print("  3. Interactive mode: python inference.py --model <model_id> --interactive")
        return

    print(f"Using model: {model_id}")

    # Run inference based on mode
    if args.interactive:
        interactive_mode(client, model_id)
    elif args.batch:
        batch_inference(client, model_id, args.batch, args.output)
    elif args.question:
        single_inference(client, model_id, args.question)
    else:
        # Default to interactive mode
        print("No mode specified. Starting interactive mode...")
        interactive_mode(client, model_id)


if __name__ == "__main__":
    main()
