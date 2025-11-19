# Evaluation Metrics for Code-Mixed Conversation Summarization

This document describes the evaluation metrics used to assess our code-mixed conversation summarization system.

## Overview

We evaluate our system using three complementary metrics that capture different aspects of summary quality:

1. **ROUGE-L**: Measures lexical overlap and n-gram matching
2. **BERTScore**: Measures semantic similarity using contextual embeddings  
3. **Code-Mixing Coverage (CMC)**: Measures language distribution shifts

## 1. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)

### Definition

ROUGE-L measures the longest common subsequence (LCS) between the generated summary and reference summary. It captures sentence-level structure similarity without requiring consecutive matches.

### Formula

```
ROUGE-L = F1-score based on LCS

Precision_LCS = LCS(generated, reference) / length(generated)
Recall_LCS = LCS(generated, reference) / length(reference)

F1-score = 2 × (Precision_LCS × Recall_LCS) / (Precision_LCS + Recall_LCS)
```

### Interpretation

- **Range**: 0.0 to 1.0 (higher is better)
- **> 0.30**: Good for abstractive summarization
- **> 0.40**: Excellent performance

### Why We Use It

ROUGE-L is the standard metric for summarization evaluation, allowing comparison with published baselines.

### References

- Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *Text Summarization Branches Out*.

---

## 2. BERTScore

### Definition

BERTScore computes similarity using contextual embeddings from BERT. It captures semantic similarity even when different words are used.

### Formula

```
BERTScore uses cosine similarity between BERT embeddings:

For tokens x in generated and y in reference:
Precision = (1/|x|) × Σ_xi max_yj (xi · yj)
Recall = (1/|y|) × Σ_yj max_xi (xi · yj)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Interpretation

- **Range**: 0.0 to 1.0 (higher is better)
- **0.75-0.85**: Good semantic similarity
- **> 0.85**: Excellent semantic similarity

### Why We Use It

BERTScore handles paraphrases and works well with multilingual/code-mixed text via multilingual BERT.

### References

- Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*.

---

## 3. Code-Mixing Coverage (CMC)

### Definition

Code-Mixing Coverage measures how well the summary preserves the language distribution of the original thread.

### Formula

```
CMC = 1 - (1/2) × Σ_languages |ratio_thread(L) - ratio_summary(L)|
```

### Interpretation

- **Range**: 0.0 to 1.0
- **Note**: For English summary generation, **low CMC is expected and acceptable**

### Why We Use It

We include CMC for transparency. Our system intentionally generates English summaries from code-mixed input, so low CMC indicates successful language normalization.

---

## Implementation

See the Colab notebook (Steps 10-11) for complete implementation:

```python
# ROUGE-L
from rouge_score import rouge_scorer

# BERTScore  
from bert_score import score as bert_score_fn

# Code-Mixing Coverage
from langdetect import detect_langs
```

---

## Running Evaluation

Evaluation is integrated into the training notebook and runs automatically after training completes. Results are saved to:

- `multi_dataset_results.json` - Summary scores
- `detailed_results.json` - Complete information
- `predictions_*.jsonl` - Generated summaries

---

## Example Command Line Usage

```bash
# If you have score.py as a separate script:
python score.py \
  --predictions predictions_cs_sum.jsonl \
  --references test_references.jsonl \
  --output scores.json
```

Example output:
```json
{
  "rougeL": 0.378,
  "bertscore_f1": 0.811,
  "code_mixing_coverage": 0.425
}
```
