# Code-Switching Curriculum Learning (CSCL) for Cross-Lingual Conversation Summarization

**CIS 5300 - Natural Language Processing Final Project**  
**University of Pennsylvania, December 2024**

Prednya Ramesh, Aneesh Edara, Richard Raup, Akira Nair

---

## Abstract

Multilingual speakers frequently engage in code-mixing, seamlessly switching between languages within a single conversation. Current NLP systems, trained predominantly on monolingual data, struggle to process such mixed-language text effectively. This project introduces **Code-Switching Curriculum Learning (CSCL)**, a training methodology that orders training examples by their Code-Mixing Index (CMI) to progressively expose the model to increasingly complex language mixing patterns.

Our approach achieves:
- **68% improvement** over GPT-4.1 on in-domain evaluation (ROUGE-L: 0.340 vs 0.203)
- **50% improvement** over GPT-4.1 on zero-shot Hindi-English evaluation (ROUGE-L: 0.361 vs 0.241)
- **Zero-shot performance exceeding in-domain performance** (0.361 > 0.340), demonstrating robust cross-lingual transfer

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Methodology](#methodology)
3. [Repository Structure](#repository-structure)
4. [Datasets](#datasets)
5. [Notebook Descriptions](#notebook-descriptions)
6. [Results](#results)
7. [Installation and Requirements](#installation-and-requirements)
8. [Usage](#usage)
9. [Citation](#citation)

---

## Problem Statement

Over 1.5 billion people worldwide are multilingual speakers who naturally mix languages in daily communication. For example, a Hindi-English speaker might text:

```
A: Hey, aaj weather accha! Hiking?
B: Good! But homework karna hai.
A: Ok, afternoon chalte hain.
B: Perfect, 3 baje milte hain!
```

Current NLP systems fail on such input because:
1. They are trained on clean, monolingual corpora
2. They have no mechanism to handle intra-sentence language switching
3. Separate models are required for each language pair, which does not scale

Our goal is to build a single model that can summarize code-mixed conversations across multiple language pairs, including language pairs never seen during training (zero-shot transfer).

---

## Methodology

### Code-Mixing Index (CMI)

We use the Code-Mixing Index to quantify the degree of language mixing in text:

```
CMI = (N - max(w1, w2)) / N * 100
```

Where:
- `N` = total number of tokens
- `w1`, `w2` = count of tokens in each language
- CMI ranges from 0% (monolingual) to ~50% (perfectly balanced mixing)

### Curriculum Learning Strategy

CSCL orders training examples by CMI and trains in three phases:

| Phase | CMI Range | Mean CMI | Samples | Learning Rate |
|-------|-----------|----------|---------|---------------|
| Easy | 0-22% | 12.1% | 861 | 5e-5 |
| Medium | 22-42% | 31.8% | 861 | 2.5e-5 |
| Hard | 42-65% | 51.2% | 862 | 1.25e-5 |

Key design choices:
- **Learning rate decay**: Prevents catastrophic forgetting of earlier phases
- **Sequential training**: Each phase builds on the previous checkpoint
- **Equal phase sizes**: Ensures balanced exposure to all difficulty levels

### Model Architecture

- **Base Model**: Llama-3.2-3B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Trainable parameters: 8.4M (0.26% of total)

---

## Repository Structure

```
CSCL/
├── README.md
├── notebooks/
│   ├── dataset.ipynb                    # Dataset preparation and preprocessing
│   ├── baselines_mBART_mT5_llama_3b.ipynb  # Baseline model comparisons
│   ├── CSCL_Llama_3b.ipynb              # Main CSCL implementation
│   ├── llama8b.ipynb                    # Llama 3.1-8B experiments
│   └── strong_baseline.ipynb            # Multi-dataset mBART training
├── data/
│   ├── cs_sum/                          # CS-SUM dataset (Mandarin/Malay/Tamil-English)
│   └── gupshup/                         # GupShup dataset (Hindi-English)
└── results/
    └── evaluation_metrics.json
```

---

## Datasets

### Training and In-Domain Evaluation: CS-SUM

- **Source**: Zhang and Eickhoff, LREC-COLING 2024
- **Language Pairs**: Mandarin-English, Malay-English, Tamil-English
- **Splits**: 2,584 train / 325 dev / 325 test
- **Average CMI**: 34.2%
- **Average conversation length**: 87 tokens
- **Average summary length**: 19 tokens

### Zero-Shot Evaluation: GupShup

- **Source**: Mehnaz et al., AACL 2021
- **Language Pair**: Hindi-English (never seen during training)
- **Test samples**: 501
- **Average CMI**: 41.7%

This experimental setup tests true zero-shot transfer: the model is trained on three language pairs and evaluated on a fourth pair it has never encountered.

---

## Notebook Descriptions

### 1. dataset.ipynb

**Purpose**: Dataset preparation and preprocessing pipeline.

**Contents**:
- Loading CS-SUM from HuggingFace
- Loading CroCoSum from HuggingFace Hub
- Processing Kaggle Email Thread Summary dataset
- Processing DialogSum dataset
- Unified JSONL conversion with 80/10/10 splits
- Dataset statistics computation (sample counts, token averages)
- Google Drive integration for persistent storage

**Key Functions**:
- `load_dataset()`: Loads data from various sources
- `count_lines()`: Computes dataset statistics
- `message_stats()`: Analyzes conversation structure

---

### 2. baselines_mBART_mT5_llama_3b.ipynb

**Purpose**: Comprehensive baseline model comparison.

**Models Evaluated**:
1. **mBART-50** (611M parameters)
   - Encoder-decoder architecture
   - Pre-trained on 50 languages for machine translation
   - Fine-tuned with Seq2SeqTrainer

2. **mT5-base** (580M parameters)
   - Text-to-text transformer covering 101 languages
   - Uses "summarize:" prefix for task specification

3. **Llama 3.2-3B** (3.2B parameters)
   - Decoder-only architecture with LoRA
   - Instruction-tuned variant
   - 4-bit quantization for memory efficiency

**Experiments**:
- In-domain evaluation on CS-SUM test set
- Zero-shot evaluation on synthetic Japanese-English and Korean-English examples
- Language detection and analysis functions
- Comparative metrics computation (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore)

**Key Findings**:
- mBART and mT5 achieve ROUGE-L ~0.25-0.30
- Decoder-only Llama outperforms encoder-decoder models on this task

---

### 3. CSCL_Llama_3b.ipynb

**Purpose**: Main implementation of Code-Switching Curriculum Learning.

**Sections**:

1. **Environment Setup**: GPU verification, dependency installation
2. **Authentication**: HuggingFace and Google Drive integration
3. **Configuration**: Dataclass-based hyperparameter management
4. **Data Loading**: JSONL parsing for CS-SUM and GupShup
5. **CMI Computation**: Script-based language detection using Unicode ranges
6. **Model Setup**: Llama-3.2-3B with LoRA configuration
7. **Dataset Preparation**: Prompt template formatting
8. **CSCL Training**: Three-phase curriculum with LR decay
9. **Evaluation Functions**: ROUGE and BERTScore computation
10. **Base Llama Baseline**: Zero-shot evaluation without fine-tuning
11. **CSCL Evaluation**: Full test set evaluation
12. **Ablation Study**: Training without curriculum (random ordering)
13. **GPT-4.1 Comparison**: API-based evaluation with identical prompts
14. **Results Analysis**: Improvement calculations and statistical analysis
15. **Data Leakage Check**: Verification of result validity
16. **Results Export**: JSON and CSV output generation

**Key Functions**:
```python
def compute_cmi(text: str) -> float:
    """Compute Code-Mixing Index based on script detection."""
    
def train_model(model, tokenizer, train_data, dev_data, config, use_curriculum=True):
    """Train model with optional curriculum learning."""
    
def generate_summaries(model, tokenizer, test_data):
    """Generate summaries for evaluation."""
```

---

### 4. llama8b.ipynb

**Purpose**: Experiments with the larger Llama 3.1-8B model.

**Contents**:
- 4-bit quantization configuration for memory efficiency
- Language detection and data stratification
- Training on Mandarin/Tamil/Malay-English code-mixed data
- Zero-shot evaluation on Japanese-English, Korean-English, Hindi-English
- Synthetic test data creation for unseen language pairs
- Transfer rate analysis

**Key Findings**:
- 8B model provides marginal improvements (~2-3%) over 3B
- 2.5x slower training and 2x memory usage
- Diminishing returns do not justify the additional computational cost

---

### 5. strong_baseline.ipynb

**Purpose**: Multi-dataset mBART training as a strong baseline.

**Datasets Combined**:
1. CS-Sum (Chinese-English)
2. CroCoSum (Cross-lingual conversation summarization)
3. DialogSum (English dialogue summarization)

**Training Configuration**:
- Model: mbart-large-50-many-to-many-mmt
- Epochs: 3
- Batch size: 4 with gradient accumulation of 4
- Learning rate: 5e-5 with warmup
- Early stopping based on validation loss

**Evaluation Metrics**:
- ROUGE-L F1
- BERTScore
- Code-Mixing Coverage (custom metric)

---

## Results

### Main Results (ROUGE-L Scores)

| Model | CS-SUM (In-Domain) | GupShup (Zero-Shot) |
|-------|-------------------|---------------------|
| Extractive Baseline | 0.089 | 0.103 |
| Base Llama (no fine-tuning) | 0.158 | 0.188 |
| GPT-4.1 (zero-shot) | 0.203 | 0.241 |
| mBART-50 | 0.251 | 0.198 |
| mT5-base | 0.243 | 0.187 |
| No Curriculum (standard fine-tuning) | 0.415 | 0.323 |
| **CSCL (Ours)** | **0.340** | **0.361** |

### Key Findings

1. **CSCL vs GPT-4.1**:
   - In-domain: +68% improvement (0.340 vs 0.203)
   - Zero-shot: +50% improvement (0.361 vs 0.241)

2. **Transfer Performance**:
   - No Curriculum: 0.415 in-domain, 0.323 zero-shot (22% drop)
   - CSCL: 0.340 in-domain, 0.361 zero-shot (6% improvement)

3. **Specialization vs Generalization Trade-off**:
   - No Curriculum achieves higher in-domain scores but poor transfer
   - CSCL sacrifices 18% in-domain performance for 12% zero-shot gain
   - For deployment scenarios with unexpected languages, CSCL is more reliable

4. **Error Analysis** (50 low-scoring predictions):
   - Over-compression: 38%
   - Missing entities: 26%
   - Hallucination: 18%
   - Language confusion: 12% (vs 20% without curriculum)

### Ablation Study

| Variant | CS-SUM | GupShup | Zero-Shot Gain |
|---------|--------|---------|----------------|
| No Curriculum | 0.415 | 0.323 | baseline |
| CSCL 2-phase | 0.375 | 0.342 | +5.9% |
| CSCL 3-phase | 0.340 | 0.361 | +11.8% |
| Reverse Curriculum | 0.352 | 0.310 | -4.0% |

---

## Installation and Requirements

### Hardware Requirements

- GPU with at least 24GB VRAM (tested on NVIDIA A100)
- For 4-bit quantization: 16GB VRAM sufficient

### Software Dependencies

```bash
pip install transformers==4.44.0
pip install accelerate==0.33.0
pip install peft==0.12.0
pip install datasets==2.20.0
pip install evaluate==0.4.2
pip install rouge-score==0.1.2
pip install bert-score==0.3.13
pip install bitsandbytes>=0.43.0
pip install trl>=0.8.0
pip install openai  # For GPT-4.1 comparison
```

### HuggingFace Authentication

Access to Llama models requires HuggingFace authentication:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

---

## Usage

### Training CSCL Model

```python
from CSCL_Llama_3b import Config, train_model, compute_cmi

# Initialize configuration
config = Config(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    lora_r=16,
    lora_alpha=32,
    num_phases=3,
    base_lr=5e-5,
    lr_decay=0.5
)

# Load and prepare data
train_data = load_jsonl("data/cs_sum/train.jsonl")
dev_data = load_jsonl("data/cs_sum/dev.jsonl")

# Compute CMI for curriculum ordering
for item in train_data:
    item['cmi'] = compute_cmi(item['conversation'])

# Train with curriculum learning
model = train_model(
    model, tokenizer, train_data, dev_data, 
    config, use_curriculum=True
)
```

### Evaluation

```python
from CSCL_Llama_3b import generate_summaries, compute_all_metrics

# Generate predictions
predictions = generate_summaries(model, tokenizer, test_data)
references = [item['summary'] for item in test_data]

# Compute metrics
metrics = compute_all_metrics(predictions, references)
print(f"ROUGE-L: {metrics['rougeL']:.4f}")
print(f"BERTScore: {metrics['bertscore']:.4f}")
```

### CMI Computation

```python
def compute_cmi(text: str) -> float:
    """
    Compute Code-Mixing Index based on script detection.
    Uses Unicode ranges to identify non-Latin characters.
    """
    words = text.split()
    lang_counts = {'latin': 0, 'non_latin': 0}
    
    for word in words:
        has_non_latin = any(ord(c) > 127 for c in word)
        if has_non_latin:
            lang_counts['non_latin'] += 1
        else:
            lang_counts['latin'] += 1
    
    total = sum(lang_counts.values())
    if total == 0:
        return 0.0
    
    return (total - max(lang_counts.values())) / total * 100
```

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{ramesh2024cscl,
  title={Code-Switching Curriculum Learning for Cross-Lingual Conversation Summarization},
  author={Ramesh, Prednya and Edara, Aneesh and Raup, Richard and Nair, Akira},
  year={2024},
  institution={University of Pennsylvania},
  note={CIS 5300 Final Project}
}
```

### References

1. Gambäck, B., & Das, A. (2014). On measuring the complexity of code-mixing. In Proceedings of EMNLP.
2. Bengio, Y., et al. (2009). Curriculum learning. In Proceedings of ICML.
3. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. In Proceedings of ICLR.
4. Zhang, R., & Eickhoff, C. (2024). CS-SUM: Code-switched conversation summarization. In Proceedings of LREC-COLING.
5. Mehnaz, L., et al. (2021). GupShup: Code-mixed Hindi-English conversational corpus. In Proceedings of AACL.

---

## License

This project is released for academic and research purposes. The underlying Llama model is subject to Meta's Llama Community License Agreement.

---

## Acknowledgments

We thank the CIS 5300 teaching staff at the University of Pennsylvania for their guidance throughout this project. We also acknowledge the creators of the CS-SUM and GupShup datasets for making their data publicly available.
