#!/usr/bin/env python3
"""
Simple Baseline: Lead-3 Extractive Summarization

This baseline extracts the first 3 speaker turns from each conversation thread
as the summary. This is a standard extractive baseline in conversation summarization.

Rationale:
- Simple and fast
- No training required
- Often captures opening context
- Serves as lower bound for more sophisticated methods

Usage:
    python simple-baseline.py --input test.jsonl --output predictions.jsonl
"""

import json
import argparse
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_lead_k(messages: List[Dict], k: int = 3) -> str:
    """
    Extract first k turns as summary.
    
    Args:
        messages: List of message dicts with 'text' field
        k: Number of turns to extract (default: 3)
    
    Returns:
        Summary string (concatenated first k messages)
    """
    # Take first k messages
    lead_messages = messages[:k]
    
    # Concatenate their text
    summary_parts = []
    for msg in lead_messages:
        text = msg.get('text', '')
        if text:
            summary_parts.append(text)
    
    # Join with space
    summary = ' '.join(summary_parts)
    
    return summary


def generate_baseline_summaries(data: List[Dict], k: int = 3) -> List[Dict]:
    """
    Generate Lead-k summaries for all threads.
    
    Args:
        data: List of thread dicts
        k: Number of turns to extract
    
    Returns:
        List of predictions with thread_id and summary
    """
    predictions = []
    
    for item in data:
        thread_id = item.get('thread_id', 'unknown')
        messages = item.get('messages', [])
        
        # Generate summary
        summary = extract_lead_k(messages, k=k)
        
        predictions.append({
            'thread_id': thread_id,
            'summary': summary
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Lead-3 extractive baseline for conversation summarization'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file with conversation threads (JSONL)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output file for predictions (JSONL)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of lead turns to extract (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} threads")
    
    # Generate summaries
    print(f"Generating Lead-{args.k} summaries...")
    predictions = generate_baseline_summaries(data, k=args.k)
    
    # Save predictions
    save_jsonl(predictions, args.output)
    print(f"Saved predictions to {args.output}")
    print(f"\nGenerated {len(predictions)} summaries")
    
    # Show example
    if predictions:
        print("\n" + "="*70)
        print("EXAMPLE OUTPUT:")
        print("="*70)
        example = predictions[0]
        print(f"Thread ID: {example['thread_id']}")
        print(f"Summary: {example['summary'][:200]}...")
        print("="*70)


if __name__ == '__main__':
    main()