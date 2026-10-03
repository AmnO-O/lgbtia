"""Evaluation metrics for Task B: 3-Class Hate Speech Classification.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


LABELS = [0, 1, 2]
HATE_CLASS_NAMES = ['no', 'yes_implicit', 'yes_explicit']


def compute_task_b_metrics(
    y_trues: np.ndarray,
    y_preds: np.ndarray,
    prefix: str = ""
) -> Dict[str, Any]:
    """Computes Macro F1, Accuracy, and per-class Precision/Recall/F1/Support for Task B.

    Guarantees fixed indexing [0: no, 1: yes_implicit, 2: yes_explicit] by setting
    labels=[0, 1, 2], preventing silent misalignment when a class is absent in a split.
    """
    y_trues = np.asarray(y_trues)
    y_preds = np.asarray(y_preds)

    acc = accuracy_score(y_trues, y_preds)
    macro_f1 = f1_score(y_trues, y_preds, labels=LABELS, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_trues, y_preds, labels=LABELS, average='weighted', zero_division=0)
    macro_precision = precision_score(y_trues, y_preds, labels=LABELS, average='macro', zero_division=0)
    macro_recall = recall_score(y_trues, y_preds, labels=LABELS, average='macro', zero_division=0)

    # Per-class metrics strictly anchored to [0, 1, 2]
    per_class_f1 = f1_score(y_trues, y_preds, labels=LABELS, average=None, zero_division=0)
    per_class_prec = precision_score(y_trues, y_preds, labels=LABELS, average=None, zero_division=0)
    per_class_rec = recall_score(y_trues, y_preds, labels=LABELS, average=None, zero_division=0)

    # Confusion matrix anchored to [0, 1, 2]
    cm = confusion_matrix(y_trues, y_preds, labels=LABELS)

    pfx = f"{prefix}_" if prefix else ""
    metrics = {
        f"{pfx}macro_f1": float(macro_f1),
        f"{pfx}accuracy": float(acc),
        f"{pfx}weighted_f1": float(weighted_f1),
        f"{pfx}macro_precision": float(macro_precision),
        f"{pfx}macro_recall": float(macro_recall),
    }

    for idx, class_name in enumerate(HATE_CLASS_NAMES):
        metrics[f"{pfx}f1_{class_name}"] = float(per_class_f1[idx])
        metrics[f"{pfx}precision_{class_name}"] = float(per_class_prec[idx])
        metrics[f"{pfx}recall_{class_name}"] = float(per_class_rec[idx])
        metrics[f"{pfx}support_{class_name}"] = int((y_trues == idx).sum())

    # Critical diagnostic confusion patterns for hierarchical analysis
    # Row 1 (True implicit):
    metrics[f"{pfx}cm_implicit_as_no"] = int(cm[1, 0])
    metrics[f"{pfx}cm_implicit_as_explicit"] = int(cm[1, 2])
    # Row 0 (True no-hate):
    metrics[f"{pfx}cm_no_as_implicit"] = int(cm[0, 1])
    metrics[f"{pfx}cm_no_as_explicit"] = int(cm[0, 2])
    # Row 2 (True explicit):
    metrics[f"{pfx}cm_explicit_as_no"] = int(cm[2, 0])
    metrics[f"{pfx}cm_explicit_as_implicit"] = int(cm[2, 1])

    return metrics


def evaluate_task_b_by_language(
    df_val: pd.DataFrame,
    y_preds: np.ndarray
) -> pd.DataFrame:
    """Computes Macro-F1, Precision, Recall, and per-class metrics across languages."""
    records = []
    df_eval = df_val.copy()
    df_eval['pred'] = y_preds

    # Overall
    overall = compute_task_b_metrics(df_eval['label'].values, df_eval['pred'].values)
    overall['language'] = 'OVERALL'
    overall['count'] = len(df_eval)
    records.append(overall)

    # By language
    if 'lang' in df_eval.columns:
        for lang, group in df_eval.groupby('lang'):
            m = compute_task_b_metrics(group['label'].values, group['pred'].values)
            m['language'] = lang
            m['count'] = len(group)
            records.append(m)

    res_df = pd.DataFrame(records)
    cols = [
        'language',
        'count',
        'macro_f1',
        'accuracy',
        'f1_no',
        'f1_yes_implicit',
        'f1_yes_explicit',
        'recall_yes_implicit',
        'recall_yes_explicit',
        'cm_implicit_as_no',
        'cm_implicit_as_explicit'
    ]
    return res_df[[c for c in cols if c in res_df.columns]]

