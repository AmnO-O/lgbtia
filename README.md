# Task B: Hate Speech Classification (StereoQueerEval)
## Pretrained Attention Weight Borrowing with K-Latent Queries (mmBERT)

A specialized PyTorch ML architecture designed for **Task B (Hate Speech Classification)** in the **StereoQueerEval** (SemEval 2027) benchmark.

---

## 🎯 Task B Overview

Task B is a **3-class hierarchical classification problem** classifying multilingual YouTube comments in video context:

| Label Index | Class Name | Description |
|---|---|---|
| `0` | **`no`** | The comment is not hateful. |
| `1` | **`yes_implicit`** | The comment conveys indirect, veiled, or implicit hate speech. |
| `2` | **`yes_explicit`** | The comment conveys direct, overt, or explicit hate speech. |

### Evaluation Metrics
- **Macro-Averaged F1** across the 3 classes (Primary competition metric).
- Accuracy & per-language performance breakdown (English, Italian, Dutch, Persian).

---

## 💡 Key Architectural Details

1. **K-Latent Pretrained Attention Borrowing**:
   - Freezes the lower layers (0–16) of a multilingual ModernBERT (`jhu-clsp/mmbert-base`) context encoder.
   - Extracts upper attention blocks (layers 17–21), repurposing their pretrained $W_Q, W_K, W_V, W_O$ projections for global cross-attention between $K$ learnable latent class queries ($Z \in \mathbb{R}^{K \times D}$) and context keys/values ($H$).
2. **Asymmetric Rotary Position Embedding (RoPE)**:
   - RoPE is applied exclusively to context Keys ($K = \text{RoPE}(H W_K)$), leaving abstract class queries ($Q = Z W_Q$) position-free.
3. **Exact Hierarchical Compound Loss**:
   - Decomposes predictions into a 2-level decision tree:
     - Level 1: Hate vs. Non-hate ($p_{\text{hate}} = \sigma(z_h)$)
     - Level 2: Implicit vs. Explicit ($p_{\text{imp}} = \sigma(z_f)$)
   - Induces compound probabilities:
     - $P(\text{no}) = 1 - p_{\text{hate}}$
     - $P(\text{implicit}) = p_{\text{hate}} \times p_{\text{imp}}$
     - $P(\text{explicit}) = p_{\text{hate}} \times (1 - p_{\text{imp}})$
   - Exactly satisfies $P(\text{no}) + P(\text{implicit}) + P(\text{explicit}) \equiv 1.0$.
4. **Data Leakage Prevention**:
   - `GroupShuffleSplit` strictly grouped by YouTube video title (`yt_title`) ensures comments from the same video are never split across train and validation sets.
5. **Multilingual Diacritic & Unicode Normalization**:
   - Uses NFC normalization to preserve critical accents (`perché`, `è`) for Italian, Dutch, and Persian.

---

## 📁 Codebase Layout

```
├── src/
│   ├── config.py       # ModelConfig and TrainingConfig dataclasses
│   ├── data.py         # TSV parser, text cleaning, HateSpeechDataset, GroupShuffleSplit
│   ├── models.py       # TransplantedKLatentModel, PretrainedCrossAttentionLayer, HierarchicalTreeHead
│   ├── losses.py       # HierarchicalCompoundLoss (exact 3-class NLL)
│   ├── metrics.py      # Macro F1, Accuracy, per-class F1, multilingual breakdown
│   ├── train.py        # Task B training loop with AMP, GradScaler, & Early Stopping
│   └── predict.py      # Inference on batches or test TSVs
├── run_pipeline.py     # Central CLI for training and inference
├── requirements.txt    # Project dependencies
├── plan.md             # Formal research hypothesis and mathematical specifications
└── README.md
```

---

## 🚀 Usage

### 1. Training the Model

Train with default K-Latent configuration ($K=8$, layer-matched context):

```bash
python run_pipeline.py --mode train \
    --model_name "jhu-clsp/mmbert-base" \
    --num_query_slots 8 \
    --context_mode layer_matched \
    --batch_size 16 \
    --epochs 10 \
    --head_lr 2e-4 \
    --output_dir ./outputs_task_b
```

Run Pre-Normalized Borrowing (Experiment B) with transplanted FFN:

```bash
python run_pipeline.py --mode train \
    --use_prenorm \
    --use_ffn \
    --num_query_slots 8 \
    --batch_size 16 \
    --output_dir ./outputs_task_b
```

### 2. Inference on Test TSV

```bash
python run_pipeline.py --mode predict \
    --checkpoint "./outputs_task_b/best_task_b_model.pt" \
    --test_tsv "StereoQueerEval_TEST.tsv" \
    --submission_tsv "./outputs_task_b/submission_task_b.tsv"
```

### 3. Python API

```python
from src.predict import HateSpeechPredictor

predictor = HateSpeechPredictor(checkpoint_path="./outputs_task_b/best_task_b_model.pt")

results = predictor.predict_batch(
    comments=["These people shouldn't be allowed in public."],
    titles=["Amsterdam Pride News"],
    descriptions=["Local report on parade events."]
)

print(results[['yt_comment', 'hate_speech_pred', 'confidence', 'prob_no', 'prob_yes_implicit', 'prob_yes_explicit']])
```
