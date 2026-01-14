#!/usr/bin/env python3
"""
Dataset preparation script for Nscale fine-tuning.

This script converts your raw data into the CSV format required by Nscale,
with 'question' and 'answer' columns.
"""

import pandas as pd
from typing import Dict, Optional, List
import json
import argparse
from pathlib import Path


def example_to_qa_row(ex: Dict) -> Optional[Dict[str, str]]:
    """
    Convert a data example to question/answer format.

    Customize this function based on your data structure.

    Args:
        ex: Dictionary containing raw data example

    Returns:
        Dictionary with 'question' and 'answer' keys, or None if invalid
    """
    question = str(ex.get("question", "")).strip()

    # Handle different answer formats
    long_answer = ex.get("long_answer")
    short_answer = ex.get("answer")
    final_decision = ex.get("final_decision")

    # Prefer long answers over short answers or decisions
    if long_answer:
        answer = str(long_answer).strip()
    elif short_answer:
        answer = str(short_answer).strip()
    elif final_decision:
        answer = str(final_decision).strip()
    else:
        answer = ""

    # Skip invalid examples
    if not question or not answer:
        return None

    return {"question": question, "answer": answer}


def load_json_data(file_path: str) -> List[Dict]:
    """Load data from a JSON or JSONL file."""
    data = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        # Try JSONL format first
        try:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        except json.JSONDecodeError:
            # If JSONL fails, try regular JSON
            f.seek(0)
            data = json.load(f)

    return data


def prepare_dataset(
    input_file: str,
    output_file: str,
    max_examples: Optional[int] = None
) -> None:
    """
    Prepare dataset from raw data file to Nscale CSV format.

    Args:
        input_file: Path to input data file (JSON or JSONL)
        output_file: Path to output CSV file
        max_examples: Maximum number of examples to process (optional)
    """
    print(f"Loading data from {input_file}...")
    raw_data = load_json_data(input_file)

    print(f"Processing {len(raw_data)} examples...")
    qa_rows = []

    for ex in raw_data[:max_examples] if max_examples else raw_data:
        qa_row = example_to_qa_row(ex)
        if qa_row:
            qa_rows.append(qa_row)

    print(f"Converted {len(qa_rows)} valid examples")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(qa_rows)

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst example:")
    print(f"Q: {df.iloc[0]['question'][:100]}...")
    print(f"A: {df.iloc[0]['answer'][:100]}...")


def create_sample_dataset(output_dir: str = "data") -> None:
    """Create sample training and validation datasets for testing."""

    # Sample QA pairs (medical/scientific domain)
    sample_data = [
        {
            "question": "What is the primary function of mitochondria in cells?",
            "answer": "Mitochondria are the powerhouse of the cell, responsible for producing ATP through cellular respiration. They convert nutrients into energy that the cell can use."
        },
        {
            "question": "What causes type 2 diabetes?",
            "answer": "Type 2 diabetes is primarily caused by insulin resistance, where cells don't respond properly to insulin. Risk factors include obesity, lack of physical activity, genetics, and age."
        },
        {
            "question": "How does the immune system respond to viral infections?",
            "answer": "The immune system responds to viral infections through both innate and adaptive immunity. It produces antibodies, activates T-cells, and creates memory cells for future protection."
        },
        {
            "question": "What is the role of DNA in protein synthesis?",
            "answer": "DNA contains the genetic instructions for making proteins. Through transcription and translation, DNA sequences are converted to mRNA and then to amino acid sequences that form proteins."
        },
        {
            "question": "What are the main symptoms of hypertension?",
            "answer": "Hypertension often has no symptoms, which is why it's called a 'silent killer'. When symptoms occur, they may include headaches, shortness of breath, nosebleeds, or chest pain."
        },
    ]

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create training set (expand sample data)
    train_data = sample_data * 10  # 50 examples
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    print(f"Created sample training set: {output_dir}/train.csv ({len(train_data)} examples)")

    # Create validation set
    val_data = sample_data[:2]  # 2 examples
    val_df = pd.DataFrame(val_data)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    print(f"Created sample validation set: {output_dir}/validation.csv ({len(val_data)} examples)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Nscale fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input data file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset for testing"
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset()
    elif args.input and args.output:
        prepare_dataset(args.input, args.output, args.max_examples)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Create sample dataset")
        print("  python prepare_dataset.py --create-sample")
        print("\n  # Convert your own data")
        print("  python prepare_dataset.py --input data.json --output data/train.csv")


if __name__ == "__main__":
    main()
