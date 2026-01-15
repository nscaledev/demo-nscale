#!/usr/bin/env python3
"""
Inference script for Nscale API.

This script provides inference capabilities using deployed models from Nscale's
inference endpoints (not fine-tuned models, which are not yet supported for inference).
"""

import os
import argparse
from dotenv import load_dotenv

from nscale import InferenceClient
from nscale.exceptions import NscaleError


def list_models_command(client: InferenceClient):
    """List available inference endpoints."""
    print("=" * 80)
    print("Available Inference Endpoints")
    print("=" * 80)

    try:
        models = client.list_models()

        if not models:
            print("No models available.")
            return

        for model in models:
            model_id = model.get("id", "N/A")
            model_type = model.get("object", "N/A")
            owned_by = model.get("owned_by", "N/A")

            print(f"\nModel ID: {model_id}")
            print(f"  Type: {model_type}")
            print(f"  Owner: {owned_by}")

        print(f"\n{len(models)} model(s) available")
        print("=" * 80)

    except NscaleError as e:
        print(f"Error listing models: {e}")


def interactive_mode(client: InferenceClient, model: str):
    """Run interactive question-answering session."""
    print("=" * 80)
    print(f"Interactive Mode - Model: {model}")
    print("=" * 80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print()

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nGenerating answer...")

            answer = client.generate_answer(
                model=model,
                question=question,
                max_tokens=1024,
                temperature=0.3,
            )

            print(f"\nAnswer: {answer}")
            print("-" * 80)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except NscaleError as e:
            print(f"\nError: {e}")
            print()


def single_question_mode(client: InferenceClient, model: str, question: str):
    """Answer a single question."""
    print("=" * 80)
    print(f"Model: {model}")
    print("=" * 80)
    print(f"Question: {question}")
    print()

    try:
        answer = client.generate_answer(
            model=model,
            question=question,
            max_tokens=1024,
            temperature=0.3,
        )

        print(f"Answer: {answer}")
        print("=" * 80)

    except NscaleError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Nscale deployed models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use for inference (e.g., 'Qwen/Qwen3-4B-Instruct-2507')"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (for non-interactive mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (continuous Q&A session)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available inference models"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature 0-2 (default: 0.3)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_token = os.getenv("NSCALE_TOKEN") or os.getenv("NSCALE_API_TOKEN")

    if not api_token:
        print("Error: NSCALE_TOKEN not found in .env file or environment.")
        print("\nPlease add your token to the .env file:")
        print("  NSCALE_TOKEN=your_token_here")
        return 1

    # Initialize client
    client = InferenceClient(api_token=api_token)

    try:
        # List models command
        if args.list_models:
            list_models_command(client)
            return 0

        # Require model for inference operations
        if not args.model:
            print("Error: No model ID provided.")
            print("\nTo get started:")
            print("  1. List available models: python inference.py --list-models")
            print("  2. Run inference: python inference.py --model <model_id> --question 'Your question'")
            print("  3. Interactive mode: python inference.py --model <model_id> --interactive")
            return 1

        # Interactive mode
        if args.interactive:
            interactive_mode(client, args.model)
            return 0

        # Single question mode
        if args.question:
            single_question_mode(client, args.model, args.question)
            return 0

        # No operation specified
        print("Error: Please specify --question or --interactive")
        print(f"\nExamples:")
        print(f"  python inference.py --model {args.model} --question 'What is AI?'")
        print(f"  python inference.py --model {args.model} --interactive")
        return 1

    except NscaleError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    exit(main())
