# Code-Mixed Conversation Summarization: Zero-Shot Cross-Lingual Transfer

## Project Overview
This project demonstrates zero-shot cross-lingual transfer for code-mixed conversation summarization. We train on 3 language pairs and test on 6 (including 3 unseen).

## Repository Structure
```
├── llama8b.ipynb
├── baselines_mBART_mT5_llama_3b.ipynb
├── milestone3_report.pdf      
├── milestone3_presentation_final.pptx  
├── README.md                     # This file
```

## Requirements
```bash
pip install transformers datasets evaluate rouge-score bert-score
pip install torch accelerate bitsandbytes
pip install peft trl
pip install pandas matplotlib seaborn tqdm
```

## How to Run

### 1. Main Model (Llama 3.1-8B)
**Hardware:** A100 GPU (40GB VRAM) recommended

```bash
# Upload to Google Colab with A100 runtime
# Run llama8b.ipynb
```

Steps:
1. Open `llama8b.ipynb` in Google Colab
2. Set runtime to GPU (A100 preferred)
3. Login to Hugging Face: `huggingface-cli login` (you also need permission to access llama models)
4. Update `DATA_DIR` path to your CS-SUM data location
5. Run all cells (~4-6 hours)

### 2. Baselines (mBART, mT5, Llama-3B)
**Hardware:** T4 GPU (16GB VRAM) sufficient

```bash
# Upload to Google Colab with T4 runtime
# Run baselines_mBART_mT5_llama_3b.ipynb
```

Steps:
1. Open `baselines_mBART_mT5_llama_3b.ipynb` in Google Colab
2. Set runtime to GPU (T4)
3. Update data paths
4. Run all cells (~6-8 hours)


## Results Summary

### Transfer Rates (Zero-Shot / In-Domain)
| Model | Transfer Rate |
|-------|---------------|
| mT5-base | 124% (failed) |
| mBART-50 | 102% |
| Llama-3.2-3B | **147%** |
| Llama-3.1-8B | 103% |

### Best Results
- **Best In-Domain:** Llama-3.1-8B (0.355 ROUGE-L)
- **Best Zero-Shot:** Llama-3.2-3B (0.437 ROUGE-L)
- **Most Consistent:** Llama-3.1-8B

## Key Files Explained

### llama8b.ipynb
- Loads Llama 3.1-8B with 4-bit quantization
- Applies LoRA fine-tuning (r=16, alpha=32)
- Trains on Chinese-En, Tamil-En, Malay-En
- Tests zero-shot on Japanese-En, Korean-En, Hindi-En
- Outputs: ROUGE, BERTScore, transfer analysis

### baselines_mBART_mT5_llama_3b.ipynb
- Trains mBART-50 (full fine-tune)
- Trains mT5-base (full fine-tune)
- Trains Llama-3.2-3B (LoRA)
- Compares all three on same test sets
- Generates comparison visualizations

## Reproducing Results

1. **Data:** Download CS-SUM from [source] or use provided splits
2. **Environment:** Google Colab with GPU
3. **Time:** ~10-14 hours total for all experiments
4. **Memory:** A100 for 8B model, T4 for others

## Presentation link
https://1drv.ms/p/c/af62c1cb85fe1248/IQCkB833x9AFQpp1bMbBEtu1AcmXMQ9syid0v__vG9c43mw?e=3VyRNR