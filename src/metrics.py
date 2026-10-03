"""Evaluation metrics for Task B: 3-Class Hate Speech Classification.
"""

from typing import Dict, List, Tuple, Optional
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


HATE_CLASS_NAMES = ['no', 'yes_implicit', 'yes_explicit']


def compute_task_b_metrics(
    y_trues: np.ndarray,
    y_preds: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Computes Macro F1, Accuracy, and per-class metrics for Task B."""
    acc = accuracy_score(y_trues, y_preds)
    macro_f1 = f1_score(y_trues, y_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_trues, y_preds, average='weighted', zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_trues, y_preds, average=None, zero_division=0)

    pfx = f"{prefix}_" if prefix else ""
    metrics = {
        f"{pfx}macro_f1": float(macro_f1),
        f"{pfx}accuracy": float(acc),
        f"{pfx}weighted_f1": float(weighted_f1)
    }

    for idx, class_name in enumerate(HATE_CLASS_NAMES):
        if idx < len(per_class_f1):
            metrics[f"{pfx}f1_{class_name}"] = float(per_class_f1[idx])

    return metrics


def evaluate_task_b_by_language(
    df_val: pd.DataFrame,
    y_preds: np.ndarray
) -> pd.DataFrame:
    """Computes Macro-F1 and Accuracy across languages (EN, IT, NL, etc.)."""
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
    cols = ['language', 'count', 'macro_f1', 'accuracy', 'f1_no', 'f1_yes_implicit', 'f1_yes_explicit']
    return res_df[[c for c in cols if c in res_df.columns]]
