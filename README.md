# Twitter Sentiment Analysis

A multi-class Twitter sentiment classification pipeline combining frozen RoBERTa embeddings with a dual-input Keras classification head fused with topic metadata, plus LIME explainability and LLM-guided counterfactual generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Project Files](#project-files)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Configuration Reference](#configuration-reference)
- [Model Architecture](#model-architecture)
- [Groq API Setup](#groq-api-setup)
- [Security Note](#security-note)
- [Suggested Improvements](#suggested-improvements)
- [Results](#results)

---

## Project Overview

This project classifies tweet sentiment across **3 classes** — `Positive`, `Neutral`, `Negative` — using:

- **Text preprocessing** — URL/mention/hashtag removal, contraction expansion, lowercasing
- **RoBERTa embeddings** — mean-pooled 768-dim vectors from `cardiffnlp/twitter-roberta-base-sentiment-latest` (PyTorch, frozen during classification head training)
- **Dual-input Keras head** — fuses text embeddings with a 32-dim learned topic embedding (12 topics), includes BatchNormalization and a skip connection
- **LIME explainability** — token-level attribution via a full re-embedding perturbation wrapper
- **Counterfactual generation** — Groq API (`llama-3.1-8b-instant`) guided by LIME word weights to produce minimal-edit sentiment-flipping rewrites

---

## Dataset

**File:** `merged_twitter_sentiment.csv`

| Property | Value |
|----------|-------|
| Total rows | 27,709 |
| Columns | `id`, `topic`, `sentiment`, `text` |
| Null values | 264 in `text` column |
| Topics | 12 (see below) |
| Raw sentiment classes | 4 (`Positive`, `Neutral`, `Negative`, `Irrelevant`) |

**Raw sentiment distribution:**

| Sentiment | Count | Share |
|-----------|-------|-------|
| Neutral | 9,153 | 33.0% |
| Negative | 8,282 | 29.9% |
| Positive | 6,423 | 23.2% |
| Irrelevant | 3,851 | 13.9% |

**Topic distribution (12 topics):**

| Topic | Count | Topic | Count |
|-------|-------|-------|-------|
| Game | 2,431 | Xbox (Xseries) | 2,360 |
| Microsoft | 2,428 | Amazon | 2,350 |
| Verizon | 2,414 | PlayStation 5 (PS5) | 2,343 |
| Facebook | 2,403 | Nvidia | 2,333 |
| Johnson & Johnson | 2,367 | HomeDepot | 2,328 |
| Google | 2,322 | Apple | 1,630 |

> **Preprocessing note:** The `Irrelevant` class and rows with null `text` are removed in Step 2 before training, leaving **23,660 samples** across 3 sentiment classes.

**Post-filtering split (70 / 15 / 15, stratified):**

| Split | Samples |
|-------|---------|
| Train | 16,562 |
| Validation | 3,549 |
| Test | 3,549 |

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Data Loading & EDA — distribution plots, null checks, sample inspection |
| 2 | Text Preprocessing & Label Encoding — cleaning, `Irrelevant` removal, 70/15/15 stratified split |
| 3 | RoBERTa Embedding Extraction & TF Dataset — mean-pooled 768-dim embeddings via PyTorch |
| 4 | Model Architecture — dual-input Keras head with skip connection and BatchNorm |
| 5 | Training — class-weighted loss, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| 6 | Evaluation — classification report, confusion matrix, training loss/accuracy curves |
| 7 | Inference Demo — single-tweet end-to-end prediction |
| 8 | LIME Explainability — token attribution with full RoBERTa re-embedding per perturbation |
| 9 | Counterfactual Generation — Groq API rewrites guided by LIME word weights, with side-by-side probability delta comparison |

---

## Project Files

```
.
├── sentiment_analysis_pipeline.ipynb   # Main notebook — full pipeline
├── merged_twitter_sentiment.csv        # Raw dataset (27,709 rows, 4 columns)
├── eda_distribution.png                # Sentiment & topic distribution bar charts
├── best_model/
│   └── model.weights.h5               # Best checkpoint saved by ModelCheckpoint
├── lime_figure/                        # LIME token attribution PNGs per demo tweet
└── loss_figure/                        # confusion_matrix.png + training_history.png
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn contractions \
            tensorflow torch scikit-learn transformers lime requests
```

> Make sure TensorFlow and PyTorch versions are compatible with your system.
> If using a GPU, verify your CUDA drivers match the installed `torch` version.

---

## How to Run

1. Place `merged_twitter_sentiment.csv` in the same directory as the notebook.
2. Open `sentiment_analysis_pipeline.ipynb` in Jupyter or VS Code.
3. Run all cells top to bottom.
4. When prompted in Step 9, enter your Groq API key.

Embedding extraction (Step 3) is the most time-consuming step. To skip it on re-runs, see [Suggested Improvements](#suggested-improvements).

---

## Configuration Reference

| Variable | Value | Description |
|----------|-------|-------------|
| `MODEL_NAME` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Pretrained RoBERTa checkpoint |
| `MAX_LEN` | `128` | Tokenizer max sequence length |
| `EMBED_BATCH_SIZE` | `64` | Batch size during embedding extraction (no gradients) |
| `BATCH_TRAIN` | `16` | Training batch size |
| `BATCH_EVAL` | `32` | Validation / test batch size |
| `ROBERTA_DIM` | `768` | RoBERTa mean-pool output dimension |
| `TOPIC_EMB_DIM` | `32` | Learned topic embedding dimension |
| `DROPOUT_RATE` | `0.4` | Dropout rate applied throughout the head |
| `EPOCHS` | `30` | Max training epochs (EarlyStopping patience = 5) |
| `SEED` | `42` | Global random seed (NumPy, TensorFlow, PyTorch) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM used for counterfactual generation |

---

## Model Architecture

```
text_embedding (768,) ──► BatchNorm ──► Dropout(0.4) ──────────────────────┐
                                                                             │
topic_id (1,) ──► Embedding(13, 32) ──► Flatten ──────────────────────► Concat (fused)
                                                                             │
                                                                       Dense(512) → BN → Dropout(0.4)
                                                                             │
                                                                       Dense(256)
                                                                             │
                                                          ┌── skip ──► Concat(256 + fused)
                                                                             │
                                                                       Dense(256) → BN → Dropout(0.4)
                                                                             │
                                                                       Dense(64, relu)
                                                                             │
                                                                       Softmax(3)
```

- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=0.01`)
- **Loss**: `SparseCategoricalCrossentropy`
- **Class balancing**: `compute_class_weight('balanced')` passed to `model.fit(class_weight=...)`
- **Callbacks**: `EarlyStopping(patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`, `ModelCheckpoint(save_weights_only=True)`

---

## Groq API Setup

Step 9 prompts for your Groq API key at runtime:

```python
API_key = input("Please enter your Groq API key: ").strip()
GROQ_API_KEY = API_key
```

For a more reproducible and secure setup, use a `.env` file:

```bash
# .env  (add this file to .gitignore)
GROQ_API_KEY=your_key_here
```

```python
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

The counterfactual generation function retries up to **3 times** on API failure before falling back to the original tweet.

---

## Security Note

- Never commit real API keys to version control.
- Add `.env` to `.gitignore`.
- Use placeholder text in any public-facing README.

---

## Suggested Improvements

- **Cache embeddings to disk** — save `train_embeddings`, `val_embeddings`, `test_embeddings` as `.npy` files after Step 3 so re-runs skip the entire RoBERTa forward pass.
- **Refactor embedding extraction** — move `extract_embeddings()` and `mean_pooling()` into a standalone `embeddings.py` script for reuse across experiments.
- **Environment-based key loading** — replace the `input()` prompt with `python-dotenv` for secure and reproducible Groq API access.
- **Expand LIME evaluation** — the current demo runs on a single hardcoded tweet; parameterize the list for systematic batch evaluation.
- **Include `Irrelevant` class** — if 4-class classification is needed, remove the filtering in Step 2.1 and update `NUM_CLASSES = 4`.

---

## Results

| Property | Detail |
|----------|--------|
| Training samples | 16,562 (post-filtering) |
| Validation samples | 3,549 |
| Test samples | 3,549 |
| Sentiment classes | 3 — `Positive`, `Neutral`, `Negative` |
| Topics | 12 |
| Text embedding dim | 768 (mean-pooled RoBERTa) |
| Topic embedding dim | 32 (learned) |

**Outputs produced:**

- `loss_figure/confusion_matrix.png` — per-class confusion matrix on the test set
- `loss_figure/training_history.png` — loss and accuracy curves across epochs
- `lime_figure/lime_step8_example_N.png` — LIME token attribution bar charts per demo tweet
- Side-by-side probability delta comparison between original and counterfactual tweets (printed in notebook output)

---

*Generated from `sentiment_analysis_pipeline.ipynb` and `merged_twitter_sentiment.csv`.*