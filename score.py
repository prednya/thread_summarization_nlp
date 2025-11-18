#!/usr/bin/env python3
"""
Evaluation script for code-mixed conversation summarization.

This script implements three metrics based on literature review:
1. ROUGE-L (Lin, 2004): Standard summarization metric
2. BERTScore (Zhang et al., 2019): Semantic similarity
3. Code-Mixing Coverage (CMC): Novel metric for bilingual faithfulness

Literature:
- Lin (2004): ROUGE: A package for automatic evaluation of summaries
- Zhang et al. (2019): BERTScore: Evaluating text generation with BERT
- Zhang & Eickhoff (2024): CroCosum - identified gap in code-mixing evaluation
- Forde et al. (2024): Recommended multilingual BERT for cross-lingual tasks

Usage:
    python score.py --predictions predictions.jsonl --references gold.jsonl
    
Output:
    Prints average scores across all examples
    
Example:
    $ python score.py --predictions mbart_test.jsonl --references cs_sum_test.jsonl
    
    ==================================================
    EVALUATION RESULTS
    ==================================================
    rougeL                        : 0.3187
    bertscore_precision           : 0.7456
    bertscore_recall              : 0.7028
    bertscore_f1                  : 0.7234
    code_mixing_coverage          : 0.6543
    ==================================================
"""

import json
import argparse
import sys
from typing import List, Dict
from collections import Counter

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("ERROR: rouge-score not installed. Run: pip install rouge-score")
    sys.exit(1)

try:
    from bert_score import score as bert_score
except ImportError:
    print("ERROR: bert-score not installed. Run: pip install bert-score")
    sys.exit(1)

try:
    from langdetect import detect_langs
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    print("ERROR: langdetect not installed. Run: pip install langdetect")
    sys.exit(1)


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    return data


def extract_text(item: Dict) -> str:
    """Extract summary text from data structure."""
    for field in ['summary', 'prediction', 'text', 'output']:
        if field in item and item[field]:
            return str(item[field])
    
    for key, value in item.items():
        if key not in ['thread_id', 'id'] and isinstance(value, str):
            return value
    
    return ""


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L F1 scores (Lin, 2004)."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)
    
    return {'rougeL': sum(scores) / len(scores) if scores else 0.0}


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore using multilingual BERT (Zhang et al., 2019)."""
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    
    if not valid_pairs:
        return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    P, R, F1 = bert_score(
        list(valid_preds), 
        list(valid_refs), 
        lang='en',
        model_type='bert-base-multilingual-cased',
        verbose=False,
        device='cpu'
    )
    
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }


def detect_language_distribution(text: str) -> Dict[str, float]:
    """Detect language distribution using word-level detection."""
    try:
        words = text.split()
        if not words:
            return {}
        
        lang_counts = Counter()
        for word in words:
            if len(word) < 3:
                continue
            try:
                langs = detect_langs(word)
                if langs:
                    lang_counts[langs[0].lang] += 1
            except LangDetectException:
                continue
        
        total = sum(lang_counts.values())
        return {lang: count / total for lang, count in lang_counts.items()} if total > 0 else {}
    except:
        return {}


def compute_code_mixing_coverage(predictions: List[str], references: List[str], threads: List[str]) -> Dict[str, float]:
    """
    Compute Code-Mixing Coverage (CMC) - Novel metric.
    
    CMC = 1 - (1/2) * Σ |lang_ratio_thread - lang_ratio_summary|
    """
    cmc_scores = []
    
    for pred, thread in zip(predictions, threads):
        if not pred or not thread:
            cmc_scores.append(0.5)
            continue
        
        thread_langs = detect_language_distribution(thread)
        pred_langs = detect_language_distribution(pred)
        
        if not thread_langs or not pred_langs:
            cmc_scores.append(0.5)
            continue
        
        all_langs = set(list(thread_langs.keys()) + list(pred_langs.keys()))
        ratio_diff = sum(abs(thread_langs.get(l, 0.0) - pred_langs.get(l, 0.0)) for l in all_langs)
        cmc = max(0.0, 1.0 - (ratio_diff / 2.0))
        cmc_scores.append(cmc)
    
    return {'code_mixing_coverage': sum(cmc_scores) / len(cmc_scores) if cmc_scores else 0.0}


def main():
    parser = argparse.ArgumentParser(description='Evaluate code-mixed conversation summarization')
    parser.add_argument('--predictions', required=True, help='Path to predictions file (JSONL)')
    parser.add_argument('--references', required=True, help='Path to gold references file (JSONL)')
    parser.add_argument('--threads', help='Path to original threads for CMC calculation (JSONL, optional)')
    parser.add_argument('--output', help='Path to save detailed results (JSON)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    pred_data = load_jsonl(args.predictions)
    ref_data = load_jsonl(args.references)
    
    predictions = [extract_text(item) for item in pred_data]
    references = [extract_text(item) for item in ref_data]
    
    if len(predictions) != len(references):
        print(f"ERROR: Mismatch: {len(predictions)} predictions vs {len(references)} references")
        sys.exit(1)
    
    print(f"Evaluating {len(predictions)} examples...\n")
    
    # Compute metrics
    print("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    
    print("Computing BERTScore...")
    bert_scores = compute_bertscore(predictions, references)
    
    cmc_scores = {}
    if args.threads:
        print("Computing Code-Mixing Coverage...")
        thread_data = load_jsonl(args.threads)
        threads = []
        for item in thread_data:
            if 'messages' in item:
                thread_text = ' '.join([msg.get('text', '') for msg in item['messages']])
            else:
                thread_text = extract_text(item)
            threads.append(thread_text)
        cmc_scores = compute_code_mixing_coverage(predictions, references, threads)
    
    all_scores = {**rouge_scores, **bert_scores, **cmc_scores}
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric in ['rougeL', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'code_mixing_coverage']:
        if metric in all_scores:
            print(f"{metric:30s}: {all_scores[metric]:.4f}")
    print("="*50 + "\n")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_scores, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()