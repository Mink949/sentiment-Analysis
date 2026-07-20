# Twitter Sentiment Analysis

A multi-class Twitter sentiment classification pipeline combining frozen RoBERTa embeddings with a dual-input Keras classification head fused with topic metadata — benchmarked against four baselines, ablated component-by-component, and paired with LIME explainability and LLM-guided counterfactual generation.

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

- **Text preprocessing** — URL/mention/hashtag removal, contraction expansion, lowercasing, plus a duplicate check and a lighter cleaning variant used for a delta comparison
- **RoBERTa embeddings** — mean-pooled 768-dim vectors from `cardiffnlp/twitter-roberta-base-sentiment-latest` (PyTorch, frozen during classification head training)
- **Dual-input Keras head** — fuses text embeddings with a 32-dim learned topic embedding (12 topics), includes BatchNormalization and a skip connection
- **Auto topic prediction, end-to-end** — a Logistic Regression topic classifier (trained on the same RoBERTa embeddings) predicts each tweet's topic *before* the main model is even built. The main model is trained and evaluated on this **predicted** topic, not the dataset's ground-truth `topic` column, so training input matches real inference — `predict_sentiment(text)` also predicts the topic first, then feeds it into the sentiment head
- **Baseline comparison (Table 3)** — zero-shot RoBERTa head, TF-IDF + Logistic Regression, TF-IDF + topic one-hot + Logistic Regression, and RoBERTa embeddings + Logistic Regression, all measured on the same test split
- **Ablation study** — the full head trained with the topic branch, skip connection, or BatchNorm removed one at a time, each at 2 epoch budgets × 5 seeds, reported as mean ± std
- **LIME explainability** — token-level attribution via a full re-embedding perturbation wrapper, explained on the exact cleaned string the model receives
- **Counterfactual generation** — Groq API (`llama-3.1-8b-instant`) guided by LIME word weights to produce a minimal-edit sentiment-flipping rewrite for a single demo tweet, with a side-by-side probability delta comparison
- **Reproducibility artifacts** — the filtered/deduplicated id list, the full run configuration, and the Groq prompt template are exported to `./artifacts/`

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
| Neutral | 9945 | 19.6% |
| Negative | 18785 | 37.0% |
| Positive | 18241 | 35.9% |
| Irrelevant | 3,851 | 7.6%% |

**Topic distribution (12 topics):**

| Topic | Count | Topic | Count |
|-------|-------|-------|-------|
| hotel | 23113 |
| League Of Legends | 2,431 | Xbox (Xseries) | 2,360 |
| Microsoft | 2,428 | Amazon | 2,350 |
| Verizon | 2,414 | PlayStation 5 (PS5) | 2,343 |
| Facebook | 2,403 | Nvidia | 2,333 |
| Johnson & Johnson | 2,367 | HomeDepot | 2,328 |
| Google | 2,322 | Apple | 1,630 |

> **Preprocessing note:** The `Irrelevant` class and rows with null `text` are removed in Step 2.1, leaving 23,660 rows. Step 2.2 cleaning empties out some rows (very short/emoji-only tweets), leaving **23,548 rows**. Step 2.2b then drops any additional rows whose `clean_text` is a duplicate of an earlier row (the aggressive cleaning collapses many distinct raw tweets — e.g. different punctuation/spacing around `"@amazon wtf"` — onto the same string), so the **final row count used for the split is lower than 23,548 and depends on the run**; the exact number and the dropped examples are printed by Step 2.2b, and the final id list is saved to `./artifacts/filtered_ids.csv`.

**Post-filtering split (70 / 15 / 15, stratified, computed *after* deduplication):**

The split is stratified on `sentiment_encoded` with `random_state=SEED`, and is recomputed on the deduplicated dataset — so exact split sizes are printed by the notebook (Step 2.5) rather than fixed here. As a reference point, prior to the Step 2.2b dedup step, the split produced 16,483 / 3,532 / 3,533 train/val/test rows; deduplication removes a further (small) number of rows from this total.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Data Loading & EDA — distribution plots, null checks, sample inspection |
| 2 | Text Preprocessing & Label Encoding — cleaning, duplicate check + removal, a light-cleaning variant, `Irrelevant` removal, 70/15/15 stratified split (with raw text carried through for later steps) |
| 3 | RoBERTa Embedding Extraction & TF Dataset — mean-pooled 768-dim embeddings via PyTorch |
| 3.5 | Topic Classifier — Logistic Regression on the RoBERTa embeddings predicts topic from text alone; its **predicted** topic (not the ground-truth `topic` column) is what feeds the Step 3.4 TF Datasets, the main model, and Baseline C below |
| 4 | Model Architecture — dual-input Keras head with skip connection and BatchNorm, plus a parameter/seed summary |
| 5 | Training — class-weighted loss, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (topic input is the Step 3.5 prediction) |
| 6 | Evaluation — classification report, confusion matrix, training loss/accuracy curves |
| 6.5 | Baseline Comparison (Table 3) — zero-shot RoBERTa head, TF-IDF+LR, TF-IDF+predicted-topic+LR, RoBERTa-embeddings+LR, and a light-cleaning delta, all vs. the main model |
| 6.6 | Ablation Study — full head vs. no-topic / no-skip / no-BatchNorm variants, 2 epoch budgets × 5 seeds, mean ± std |
| 7 | Inference Demo — single-tweet end-to-end prediction; topic is auto-predicted (Step 3.5), then fed into the sentiment head |
| 8 | LIME Explainability — token attribution with full RoBERTa re-embedding per perturbation, explained on the model's actual cleaned input, with the topic auto-predicted once and held fixed across perturbations |
| 9 | Counterfactual Generation — Groq API rewrite of one demo tweet, guided by LIME word weights, with a side-by-side probability delta comparison |
| 10 | Reproducibility Artifacts Export — filtered/deduplicated id list, run configuration, and Groq prompt template saved to `./artifacts/` |

---

## Project Files

```
.
├── sentiment_analysis_pipeline.ipynb   # Main notebook — full pipeline
├── merged_twitter_sentiment.csv        # Raw dataset (27,709 rows, 4 columns)
├── eda_distribution.png                # Sentiment & topic distribution bar charts
├── best_model/
│   └── model.weights.h5               # Best checkpoint saved by ModelCheckpoint
├── ablation_ckpt/                      # Per-run checkpoints written by the Step 6.6 ablation sweep
├── artifacts/                          # Reproducibility & results exports (Steps 2.2b, 6.5, 6.6, 10)
│   ├── filtered_ids.csv               #   ids used for the train/val/test split (post-dedup)
│   ├── baseline_comparison_table3.csv #   Table 3: baselines A-E + main model
│   ├── ablation_raw_results.csv       #   one row per (variant, epochs, seed) run
│   ├── ablation_summary_table.csv     #   mean ± std per (variant, epochs)
│   ├── run_config.json                #   all hyperparameters for this run
│   └── groq_prompts.json              #   the Groq prompt template used in Step 9
├── lime_figure/                        # LIME token attribution PNGs per demo tweet
└── loss_figure/                        # confusion_matrix.png + training_history.png
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn contractions scipy \
            tensorflow torch scikit-learn transformers lime requests
```

> Make sure TensorFlow and PyTorch versions are compatible with your system.
> If using a GPU, verify your CUDA drivers match the installed `torch` version.
> `scipy` is required directly for `scipy.sparse.hstack` used in Baseline C (Step 6.5).

---

## Topic Prediction Feeds the Main Model (not the ground-truth label)

An earlier version of this pipeline fed the dataset's ground-truth `topic` column directly into the
main model's `topic_id` input — both during training and at inference. That's unrealistic: a
deployed model doesn't get a hand-labeled topic for free. Step 3.5 now trains a topic classifier
*before* the main model exists, and its **predictions** (`topic_train_pred` / `topic_val_pred` /
`topic_test_pred`) — not `topic_train` / `topic_val` / `topic_test` — are what populate the Step 3.4
TF Datasets that Step 4's model and the Step 6.6 ablation sweep both train on. Baseline C (Step 6.5)
was updated the same way, so it stays a fair "same input as the main model" comparison. The
ground-truth `topic_train`/`topic_val`/`topic_test` arrays still exist (from the Step 2.4 split) and
are only used to *fit and score* the Step 3.5 classifier — nothing downstream sees them again.

---

## How to Run

1. Place `merged_twitter_sentiment.csv` in the same directory as the notebook.
2. Open `sentiment_analysis_pipeline.ipynb` in Jupyter or VS Code.
3. Run all cells top to bottom.
4. When prompted in Step 9, enter your Groq API key (only needed for the single-tweet counterfactual demo — no large batch of API calls is made).

Embedding extraction (Step 3) is the most time-consuming step. Step 6.6 (ablation study) is the second most expensive — it trains 40 head-only models (4 variants × 2 epoch budgets × 5 seeds) on the *cached* embeddings, so it doesn't re-run RoBERTa but is still the longest-running cell in the notebook; reduce `N_SEEDS` in that cell for a quick smoke test. To skip Step 3 entirely on re-runs, see [Suggested Improvements](#suggested-improvements).

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
| `EPOCHS` | `30` | Max epochs for the **full** model (EarlyStopping patience = 5) |
| `SEED` | `42` | Global random seed (NumPy, TensorFlow, PyTorch) — used for data splits and the main model |
| `ABLATION_VARIANTS` | `['full', 'no_topic', 'no_skip', 'no_bn']` | Architecture variants compared in Step 6.6 |
| `EPOCH_BUDGETS` | `[15, 30]` | Epoch caps swept per ablation variant |
| `N_SEEDS` | `5` | Seeds per (variant, epoch budget) cell in the ablation sweep |
| Ablation patience | `4` at 15 epochs, `5` at 30 epochs | Set via `make_callbacks(patience=...)`, proportional to the epoch budget |
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
- **Callbacks**: built by `make_callbacks(patience, checkpoint_path)` — `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`, `ModelCheckpoint(save_weights_only=True)`. The full model uses `patience=5`; the Step 6.6 ablation runs use `patience=4` at the 15-epoch budget and `patience=5` at the 30-epoch budget.
- **Ablation variants** (`build_model_ablation()`, Step 6.6): `no_topic` zero-gates the topic embedding's contribution before fusion (kept in the graph so the model signature is unchanged, but it carries no signal and gets no gradient); `no_skip` drops the `Concat(256 + fused)` skip connection; `no_bn` removes every `BatchNormalization` layer.
- **`topic_id` is a prediction, not a label**: the diagram's `topic_id` input comes from the Step 3.5 topic classifier both during training (`topic_train_pred`/`topic_val_pred`/`topic_test_pred` populate the Step 3.4 TF Datasets) and at inference. `predict_sentiment(text, topic_str=None)` (Step 7) extracts the tweet's RoBERTa embedding once, feeds it to the Step 3.5 classifier to get `topic_id`, and only then runs the sentiment head — `topic_str` can still be passed explicitly to override the auto-prediction (e.g. to check how much the model relies on topic-classifier noise vs. a clean label).

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

The counterfactual generation function retries up to **3 times** on API failure before falling back to the original tweet. Step 9 only rewrites a single demo tweet, so this stays well within Groq's free-tier rate limits; the prompt template it uses is also saved to `./artifacts/groq_prompts.json` in Step 10 for reproducibility.

---

## Security Note

- Never commit real API keys to version control.
- Add `.env` to `.gitignore`.
- Use placeholder text in any public-facing README.

---

## Suggested Improvements

- **Cache embeddings to disk** — save `train_embeddings`, `val_embeddings`, `test_embeddings` as `.npy` files after Step 3 so re-runs skip the entire RoBERTa forward pass (this would also speed up re-running Step 6.5's Baseline D and the Step 6.6 ablation sweep, both of which reuse these arrays).
- **Refactor embedding extraction** — move `extract_embeddings()` and `mean_pooling()` into a standalone `embeddings.py` script for reuse across experiments.
- **Environment-based key loading** — replace the `input()` prompt with `python-dotenv` for secure and reproducible Groq API access.
- **Expand LIME evaluation** — the current demo runs on a single hardcoded tweet; parameterize the list for systematic batch evaluation.
- **Include `Irrelevant` class** — if 4-class classification is needed, remove the filtering in Step 2.1 and update `NUM_CLASSES = 4`.
- **Batch counterfactual evaluation at scale** — Step 9 only illustrates the LIME-guided rewrite on one tweet; running it over many sampled tweets to measure a sentiment-flip rate was intentionally left out of this notebook because it would issue a very large number of Groq API calls and run into rate limits — worth doing as a separate, throttled batch job (with backoff/queuing) rather than inline in the notebook.

---

## Results

| Property | Detail |
|----------|--------|
| Training samples | ~16,483 pre-dedup; the exact post-dedup count is printed by Step 2.5 and saved to `./artifacts/filtered_ids.csv` |
| Validation / Test samples | ~3,532 / ~3,533 pre-dedup each; exact counts printed by Step 2.5 |
| Sentiment classes | 3 — `Positive`, `Neutral`, `Negative` |
| Topics | 12 |
| Text embedding dim | 768 (mean-pooled RoBERTa) |
| Topic embedding dim | 32 (learned) |

**Outputs produced:**

- `loss_figure/confusion_matrix.png` — per-class confusion matrix on the test set
- `loss_figure/training_history.png` — loss and accuracy curves across epochs
- `lime_figure/lime_step8_example_N.png` — LIME token attribution bar charts per demo tweet
- `artifacts/baseline_comparison_table3.csv` — Table 3: zero-shot RoBERTa, 3 TF-IDF/embedding + Logistic Regression baselines, and the main model, all on the same test split
- `artifacts/ablation_summary_table.csv` — mean ± std accuracy/macro-F1 for the full model vs. no-topic / no-skip / no-BatchNorm variants, at 15 and 30 epochs
- `artifacts/run_config.json`, `artifacts/groq_prompts.json`, `artifacts/filtered_ids.csv` — reproducibility artifacts (Step 10)
- Side-by-side probability delta comparison between the original and counterfactual demo tweet (printed in notebook output)

---

*Generated from `sentiment_analysis_pipeline.ipynb` and `merged_twitter_sentiment.csv`.*
