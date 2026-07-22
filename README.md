# Twitter Sentiment Analysis

A multi-class Twitter sentiment classification pipeline combining frozen RoBERTa embeddings with a dual-input Keras classification head fused with topic metadata — benchmarked against five baselines, ablated component-by-component, and paired with LIME explainability and LLM-guided counterfactual generation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Project Files](#project-files)
- [Requirements](#requirements)
- [Topic Prediction Feeds the Main Model](#topic-prediction-feeds-the-main-model-not-the-ground-truth-label)
- [How to Run](#how-to-run)
- [Configuration Reference](#configuration-reference)
- [Model Architecture](#model-architecture)
- [Groq API Setup](#groq-api-setup)
- [Security Note](#security-note)
- [Suggested Improvements](#suggested-improvements)
- [Results](#results)

---

## Project Overview

This project classifies tweet sentiment across **3 classes** — `Negative`, `Neutral`, `Positive` — using:

- **Text preprocessing** — URL/mention/hashtag removal, contraction expansion, lowercasing, plus an ambiguous-label duplicate check and a lighter cleaning variant used for a delta comparison
- **RoBERTa embeddings** — mean-pooled 768-dim vectors from `cardiffnlp/twitter-roberta-base-sentiment-latest` (PyTorch, frozen during classification head training)
- **Dual-input Keras head** — fuses text embeddings with a 32-dim learned topic embedding (13 topics), includes BatchNormalization at every dense stage and a skip connection
- **Auto topic prediction, end-to-end** — a TF-IDF + Logistic Regression topic classifier predicts each tweet's topic *before* the main model is even built. The main model is trained and evaluated on this **predicted** topic, not the dataset's ground-truth `topic` column, so training input matches real inference — `predict_sentiment(text)` also predicts the topic first, then feeds it into the sentiment head
- **Baseline comparison (Table 3)** — zero-shot RoBERTa head, TF-IDF + Logistic Regression, TF-IDF + predicted-topic one-hot + Logistic Regression, RoBERTa embeddings + Logistic Regression, and a light-cleaning delta, all measured on the same test split
- **Ablation study** — the full head trained with the topic branch, skip connection, or BatchNorm removed one at a time, each at 2 epoch budgets × 5 seeds, reported as mean ± std
- **LIME explainability** — token-level attribution via a full re-embedding perturbation wrapper, explained on the exact cleaned string the model receives
- **Counterfactual generation** — Groq API (`llama-3.1-8b-instant`) guided by LIME word weights to produce a minimal-edit sentiment-flipping rewrite for a demo tweet, with a side-by-side probability delta comparison and a saved explanation file
- **Reproducibility artifacts** — the filtered/deduplicated id list, the full run configuration, and the Groq prompt template are exported to `./artifacts/`

---

## Dataset

**File:** `merged_twitter_sentiment.csv` (checked into this repo as `merged_twitter_sentiment.xlsx` — export it to CSV before running; see [How to Run](#how-to-run))

| Property | Value |
|----------|-------|
| Total rows | 50,822 |
| Columns | `id`, `topic`, `sentiment`, `text` |
| Null values | 264 in `text` column |
| Topics | 13 (see below) |
| Raw sentiment classes | 4 (`Positive`, `Neutral`, `Negative`, `Irrelevant`) |

**Raw sentiment distribution:**

| Sentiment | Count | Share |
|-----------|-------|-------|
| Negative | 18,785 | 37.0% |
| Positive | 18,241 | 35.9% |
| Neutral | 9,945 | 19.6% |
| Irrelevant | 3,851 | 7.6% |

**Topic distribution (13 topics):**

| Topic | Count | Topic | Count |
|-------|-------|-------|-------|
| hotel | 23,113 | PlayStation5 (PS5) | 2,343 |
| League Of Legends | 2,431 | Nvidia | 2,333 |
| Microsoft | 2,428 | HomeDepot | 2,328 |
| Verizon | 2,414 | Google | 2,322 |
| Facebook | 2,403 | Apple | 1,630 |
| johnson&johnson | 2,367 | | |
| Xbox (Xseries) | 2,360 | | |
| Amazon | 2,350 | | |

> `hotel` alone accounts for 45.5% of all rows — almost as many as the other 12 topics combined (27,709). Any topic-conditioned metric should be read with this imbalance in mind.

**Preprocessing funnel (Step 2, exact counts from the committed run):**

| Stage | Rows remaining |
|-------|-----------------|
| Raw dataset | 50,822 |
| After removing `Irrelevant` | 46,971 |
| After dropping null `text`/`sentiment` | 46,773 |
| After `clean_text()` (drops rows that clean to empty) | 46,661 |
| After dropping ambiguous-label `clean_text` groups (−778) | 45,883 |
| After deduplicating remaining `clean_text` groups (−1,487) | **44,396** |

The final 44,396 ids are the ones actually used for the train/val/test split, saved to `./artifacts/filtered_ids.csv`.

**Split (70 / 15 / 15, stratified on `sentiment_encoded`, `random_state=SEED`):**

| Split | Samples | Negative | Neutral | Positive |
|-------|---------|----------|---------|----------|
| Train | 31,077 | 12,601 (40.5%) | 6,369 (20.5%) | 12,107 (39.0%) |
| Val | 6,659 | 2,700 (40.5%) | 1,365 (20.5%) | 2,594 (39.0%) |
| Test | 6,660 | 2,700 (40.5%) | 1,365 (20.5%) | 2,595 (39.0%) |

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Data Loading & EDA — distribution plots, null checks, sample inspection |
| 2 | Text Preprocessing & Label Encoding — cleaning, ambiguous-label + duplicate removal, a light-cleaning variant, `Irrelevant` removal, 70/15/15 stratified split (raw text carried through for later steps) |
| 3 | RoBERTa Embedding Extraction & TF Dataset — mean-pooled 768-dim embeddings via PyTorch |
| 3.4 | Topic Classifier — TF-IDF(1–2gram) + Logistic Regression (grid-searched) predicts topic from text alone; its **predicted** topic (not the ground-truth `topic` column) is what feeds the Step 3.5 TF Datasets, the main model, and Baseline C below |
| 4 | Model Architecture — dual-input Keras head with skip connection and BatchNorm, plus a parameter/seed summary |
| 5 | Training — class-weighted loss, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (topic input is the Step 3.4 prediction) |
| 6 | Evaluation — classification report, confusion matrix, training loss/accuracy curves |
| 6.5 | Baseline Comparison (Table 3) — zero-shot RoBERTa head, TF-IDF+LR, TF-IDF+predicted-topic+LR, RoBERTa-embeddings+LR, and a light-cleaning delta, all vs. the main model |
| 6.6 | Ablation Study — full head vs. no-topic / no-skip / no-BatchNorm variants, 2 epoch budgets × 5 seeds, mean ± std |
| 7 | Inference Demo — single-tweet end-to-end prediction; topic is auto-predicted, then fed into the sentiment head |
| 8 | LIME Explainability — token attribution with full RoBERTa re-embedding per perturbation, explained on the model's actual cleaned input, with the topic auto-predicted once and held fixed across perturbations |
| 9 | Counterfactual Generation — Groq API rewrite of a demo tweet guided by LIME word weights, re-prediction with a side-by-side probability delta comparison, a human-readable explanation, and a saved `.txt` report |
| 10 | Reproducibility Artifacts Export — run configuration and Groq prompt template saved to `./artifacts/` (the filtered/deduplicated id list was already saved in Step 2) |

---

## Project Files

```
.
├── sentiment_analysis_pipeline.ipynb   # Main notebook — full pipeline
├── merged_twitter_sentiment.csv       # Raw dataset (50,822 rows, 4 columns)
├── eda_distribution.png                # Sentiment & topic distribution bar charts
├── best_model/
│   └── model.weights.h5               # Best checkpoint saved by ModelCheckpoint
├── ablation_ckpt/                      # Per-run checkpoints from the Step 6.6 ablation sweep (40 files: 4 variants × 2 epoch budgets × 5 seeds)
├── artifacts/                          # Reproducibility & results exports (Steps 2, 6.5, 6.6, 10)
│   ├── filtered_ids.csv               #   44,396 ids used for the train/val/test split (post-dedup)
│   ├── baseline_comparison_table3.csv #   Table 3: baselines A-E + main model
│   ├── ablation_raw_results.csv       #   one row per (variant, epochs, seed) run
│   ├── ablation_summary_table.csv     #   mean ± std per (variant, epochs)
│   ├── run_config.json                #   all hyperparameters for this run
│   └── groq_prompts.json              #   the Groq prompt template used in Step 9
├── lime_figure/                        # LIME token attribution PNGs per demo tweet
├── loss_figure/                        # confusion_matrix.png + training_history.png
└── counterfactual_Example/             # Saved human-readable counterfactual explanation(s) from Step 9
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
> The dataset ships as `merged_twitter_sentiment.xlsx`; if you convert it with pandas locally you'll also need `openpyxl` (`pip install openpyxl`) purely for that one-off export, not for the notebook itself.

---

## Topic Prediction Feeds the Main Model (not the ground-truth label)

An earlier version of this pipeline fed the dataset's ground-truth `topic` column directly into the
main model's `topic_id` input — both during training and at inference. That's unrealistic: a
deployed model doesn't get a hand-labeled topic for free. Step 3.4 now trains a topic classifier
*before* the main model exists, and its **predictions** (`topic_train_pred` / `topic_val_pred` /
`topic_test_pred`) — not `topic_train` / `topic_val` / `topic_test` — are what populate the Step 3.5
TF Datasets that Step 4's model and the Step 6.6 ablation sweep both train on. Baseline C (Step 6.5)
was updated the same way, so it stays a fair "same input as the main model" comparison. The
ground-truth `topic_train`/`topic_val`/`topic_test` arrays still exist (from the Step 2 split) and
are only used to *fit and score* the Step 3.4 classifier — nothing downstream sees them again.

---

## How to Run

1. Open `sentiment_analysis_pipeline.ipynb` in Jupyter or VS Code.
2. Run all cells top to bottom.
3. When prompted in Step 9, enter your Groq API key (only needed for the single-tweet counterfactual demo — no large batch of API calls is made).

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
| `EPOCHS` | `30` | Max epochs for the **full** model (EarlyStopping patience = 5; this run used the full budget) |
| `SEED` | `42` | Global random seed (NumPy, TensorFlow, PyTorch) — used for data splits and the main model |
| `NUM_TOPICS` | `13` | Distinct topics in the deduplicated dataset (`Embedding` layer uses `input_dim=NUM_TOPICS+1=14`) |
| `NUM_CLASSES` | `3` | `Negative` / `Neutral` / `Positive` |
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
topic_id (1,) ──► Embedding(14, 32) ──► Flatten ──────────────────────► Concat (fused, 800-dim)
                                                                             │
                                                                       Dense(512) → BN → Dropout(0.4)
                                                                             │
                                                                       Dense(256)
                                                                             │
                                                          ┌── skip ──► Concat(256 + fused = 1056-dim)
                                                                             │
                                                                       Dense(256) → BN → Dropout(0.4)
                                                                             │
                                                                       Dense(64, relu)
                                                                             │
                                                                       Softmax(3)
```

- **Params**: 835,267 total — 832,195 trainable, 3,072 non-trainable (the two BatchNorm layers' running mean/variance)
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=0.01`)
- **Loss**: `SparseCategoricalCrossentropy`
- **Class balancing**: `compute_class_weight('balanced')` passed to `model.fit(class_weight=...)`
- **Callbacks**: built by `make_callbacks(patience, checkpoint_path)` — `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`, `ModelCheckpoint(save_weights_only=True)`. The full model uses `patience=5`; the Step 6.6 ablation runs use `patience=4` at the 15-epoch budget and `patience=5` at the 30-epoch budget.
- **Ablation variants** (`build_model_ablation()`, Step 6.6): `no_topic` zero-gates the topic embedding's contribution before fusion (kept in the graph so the model signature is unchanged, but it carries no signal and gets no gradient); `no_skip` drops the `Concat(256 + fused)` skip connection; `no_bn` removes every `BatchNormalization` layer.
- **`topic_id` is a prediction, not a label**: the diagram's `topic_id` input comes from the Step 3.4 topic classifier both during training (`topic_train_pred`/`topic_val_pred`/`topic_test_pred` populate the Step 3.5 TF Datasets) and at inference. `predict_sentiment(text, topic_str=None)` (Step 7) extracts the tweet's RoBERTa embedding once, feeds it to the Step 3.4 classifier to get `topic_id`, and only then runs the sentiment head — `topic_str` can still be passed explicitly to override the auto-prediction (e.g. to check how much the model relies on topic-classifier noise vs. a clean label).

---

## Groq API Setup

Step 9 prompts for your Groq API key at runtime:

```python
API_key = input('Enter Your API Key: ').strip()
GROQ_API_KEY = API_key
```

Before generating the counterfactual, the cell also lists the Groq models available on your account and makes a one-token test call (`"Say OK only."`) to confirm the key works, so a bad key fails fast instead of mid-prompt.

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

- **Commit the CSV, not just the XLSX** — the notebook hardcodes `pd.read_csv('merged_twitter_sentiment.csv')`, but only `merged_twitter_sentiment.xlsx` is checked into the repo. Either commit the CSV directly or add a one-time conversion cell so a fresh clone runs without a manual export step.
- **Cache embeddings to disk** — save `train_embeddings`, `val_embeddings`, `test_embeddings` as `.npy` files after Step 3 so re-runs skip the entire RoBERTa forward pass (this would also speed up re-running Step 6.5's Baseline D and the Step 6.6 ablation sweep, both of which reuse these arrays).
- **Refactor embedding extraction** — move `extract_embeddings()` and `mean_pooling()` into a standalone `embeddings.py` script for reuse across experiments.
- **Environment-based key loading** — replace the `input()` prompt with `python-dotenv` for secure and reproducible Groq API access.
- **Expand LIME and counterfactual evaluation** — both currently run on a single hardcoded tweet; parameterize the list for systematic batch evaluation. A batch sentiment-flip-rate experiment over many sampled tweets was intentionally left out of the notebook because it would issue a very large number of Groq API calls and hit rate limits — worth doing as a separate, throttled job (with backoff/queuing) rather than inline in the notebook.
- **Include `Irrelevant` class** — if 4-class classification is needed, remove the filtering in Step 2 and update `NUM_CLASSES = 4`.
- **Move large binaries out of plain git** — `ablation_ckpt/` (40 `.weights.h5` files), `best_model/`, and the dataset are all committed directly rather than via Git LFS; worth revisiting once the repo grows further.
- **Clean up leftover copy from earlier iterations** — a few printed strings (e.g. "Ollama Counterfactual Generation", a stray "Step 6.7" reference in the Step 7 demo) are carried over from an earlier version of the notebook that used a different LLM/step numbering; harmless at runtime but worth fixing for clarity.

---

## Results

**Split sizes:** 31,077 train / 6,659 val / 6,660 test (see [Dataset](#dataset)).

**Main model — test set classification report (Step 6):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.9181 | 0.8756 | 0.8963 | 2,700 |
| Neutral | 0.7214 | 0.8630 | 0.7859 | 1,365 |
| Positive | 0.9233 | 0.8724 | 0.8972 | 2,595 |
| **Accuracy** | | | **0.8718** | 6,660 |
| **Macro avg** | 0.8543 | 0.8703 | **0.8598** | 6,660 |

Training ran the full 30-epoch budget without early stopping (best `val_loss` 0.3265 at the final epoch, restored from there); final train accuracy 0.8771 / val accuracy 0.8761.

**Baseline comparison — Table 3 (test set, sorted by macro-F1):**

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Main model (RoBERTa emb + topic + BN + skip head) | 0.8718 | 0.8598 | trained head, Step 4–6 |
| E: TF-IDF (light cleaning) + LogisticRegression | 0.8601 | 0.8496 | keeps digits & hashtag words |
| B: TF-IDF + LogisticRegression | 0.8595 | 0.8486 | vocab = 20,000 |
| C: TF-IDF + predicted-topic one-hot + LogisticRegression | 0.8547 | 0.8443 | same (text, predicted-topic) input as the main model |
| D: RoBERTa embeddings + LogisticRegression | 0.8116 | 0.7949 | StandardScaler + max_iter=2000 |
| A: Zero-shot RoBERTa head | 0.6634 | 0.6256 | no training, model's own head |

The trained fusion head beats every baseline, but the margin over the plain TF-IDF+LR baselines (B/C/E) is only ~1–1.5pp — most of the discriminative signal in this dataset is lexical. Zero-shot RoBERTa (A) lags far behind everything trained on this data, and frozen RoBERTa embeddings fed to a linear classifier (D) actually underperform simple TF-IDF, suggesting the sentiment-tuned embedding space isn't linearly separable enough on its own without the trained head.

**Ablation study — mean ± std over 5 seeds (Step 6.6):**

| Variant | 15 epochs (acc / macro-F1) | 30 epochs (acc / macro-F1) |
|---------|---------------------------|------------------------------|
| full | 0.8443 ± 0.0043 / 0.8299 ± 0.0043 | 0.8685 ± 0.0051 / 0.8558 ± 0.0055 |
| no_skip | 0.8447 ± 0.0051 / 0.8293 ± 0.0054 | 0.8687 ± 0.0040 / 0.8554 ± 0.0042 |
| no_topic | 0.8397 ± 0.0075 / 0.8244 ± 0.0076 | 0.8593 ± 0.0033 / 0.8461 ± 0.0039 |
| no_bn | 0.8306 ± 0.0043 / 0.8152 ± 0.0041 | 0.8486 ± 0.0027 / 0.8343 ± 0.0027 |

- **BatchNorm matters most** — removing it costs ~2pp accuracy and ~2pp macro-F1 at both budgets, the largest drop of any variant.
- **The skip connection contributes essentially nothing** at 30 epochs (`no_skip` is statistically indistinguishable from `full`, and even marginally ahead within noise) — it's not pulling its weight in the current architecture.
- **The topic branch gives a real, moderate lift** — dropping it costs ~0.9pp accuracy / ~1pp macro-F1 at 30 epochs.
- More training (15 → 30 epochs) helps every variant, with `full` and `no_skip` benefiting most.

**Auxiliary topic classifier (Step 3.4):** 5-fold CV accuracy 0.9490 (`C=10.0`, `max_features=20000`); test accuracy 0.9544, test macro-F1 0.9212 — this is the classifier whose *predictions* feed the main model's `topic_id` input.

**Counterfactual demo (Step 9):** a League-of-Legends complaint tweet predicted `Negative` (83.24% confidence) was rewritten by Groq to flip to `Positive` (50.26% confidence) — probability deltas of Negative −0.8190, Neutral +0.3627, Positive +0.4563. Full explanation saved to `counterfactual_Example/counterfactual_example_1.txt`.

**Outputs produced:**

- `eda_distribution.png` — sentiment & topic distribution bar charts (Step 1)
- `loss_figure/confusion_matrix.png` — per-class confusion matrix on the test set
- `loss_figure/training_history.png` — loss and accuracy curves across epochs
- `lime_figure/lime_step8_example_1.png` — LIME token attribution bar chart for the demo tweet
- `artifacts/baseline_comparison_table3.csv` — Table 3 above
- `artifacts/ablation_summary_table.csv` / `artifacts/ablation_raw_results.csv` — ablation results, summarized and per-run
- `artifacts/run_config.json`, `artifacts/groq_prompts.json`, `artifacts/filtered_ids.csv` — reproducibility artifacts (Step 10)
- `counterfactual_Example/counterfactual_example_1.txt` — human-readable counterfactual explanation

---

*Generated from `sentiment_analysis_pipeline.ipynb` and `merged_twitter_sentiment.xlsx`.*
