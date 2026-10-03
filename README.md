# Task B: Hate Speech Classification (StereoQueerEval)

A streamlined, high-performance PyTorch NLP pipeline specifically designed for **Task B (Hate Speech Detection)** in the **StereoQueerEval** (SemEval 2027) benchmark.

---

## 🎯 Task B Overview

Task B is a **3-class classification problem** that classifies YouTube comments in context into:

| Label Index | Class Name | Description |
|---|---|---|
| `0` | **`no`** | The comment is not hateful. |
| `1` | **`yes_implicit`** | The comment conveys indirect, veiled, or implicit hate speech. |
| `2` | **`yes_explicit`** | The comment conveys direct, overt, or explicit hate speech. |

### Evaluation Metric
- **Macro-Averaged F1** ($\text{Macro-F1}$) across the 3 classes.
- Accuracy & per-language performance (English, Italian, Dutch, Persian).

---

## 💡 Key Architectural Details

1. **Context-Aware Encoding**:
   Input sequences are formatted as:
   `"Comment: <yt_comment> [SEP] Video: <yt_title> [SEP] Description: <yt_description>"`
2. **Data Leakage Prevention**:
   Validation splits are strictly grouped by video (`yt_title`) so the model never evaluates on comments belonging to videos seen during training.
3. **Multilingual Diacritic & Unicode Normalization**:
   Preserves accents (`perché`, `è`) for Italian and Dutch.
4. **Class Imbalance Management**:
   Automated compute of class weights or optional Multi-Class Focal Loss to boost minority classes (`yes_implicit` & `yes_explicit`).
5. **State-of-the-Art Multilingual Backbone**:
   Supports `microsoft/mdeberta-v3-base`, `xlm-roberta-base`, and `jhu-clsp/mmbert-base` (ModernBERT) with differential learning rates and warmup.

---

## 📁 Codebase Layout

```
├── src/
│   ├── config.py       # Configuration (ModelConfig, TrainingConfig)
│   ├── data.py         # TSV parser, text cleaning, HateSpeechDataset, Group split
│   ├── models.py       # HateSpeechClassifier (Backbone + Pooler + 3-Class Head)
│   ├── losses.py       # Weighted CrossEntropy & MultiClassFocalLoss
│   ├── metrics.py      # Macro F1, Accuracy, per-class F1, language breakdown
│   ├── train.py        # Task B training loop with AMP & Early Stopping
│   └── predict.py      # Inference on batches or test TSVs
├── run_pipeline.py     # Clean CLI for training and inference
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Train Task B Model

```bash
python run_pipeline.py --mode train \
    --model_name "microsoft/mdeberta-v3-base" \
    --batch_size 16 \
    --epochs 8 \
    --lr 2e-5 \
    --output_dir ./outputs_task_b
```

### 2. Run Inference on Test TSV

```bash
python run_pipeline.py --mode predict \
    --checkpoint "./outputs_task_b/best_task_b_model.pt" \
    --test_tsv "StereoQueerEval_TEST.tsv" \
    --submission_tsv "./outputs_task_b/submission_task_b.tsv"
```

### 3. Python API Usage

```python
from src.predict import HateSpeechPredictor

predictor = HateSpeechPredictor(checkpoint_path="./outputs_task_b/best_task_b_model.pt")

results = predictor.predict_batch(
    comments=["These people are disgusting and shouldn't exist."],
    titles=["Pride Parade in Amsterdam"],
    descriptions=["News report on pride events."]
)

print(results[['yt_comment', 'hate_speech_pred', 'confidence', 'prob_no', 'prob_yes_implicit', 'prob_yes_explicit']])
```
