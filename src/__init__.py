"""StereoQueerEval Task B (Pretrained Attention Weight Borrowing & Hierarchical Loss).
"""

from src.config import ModelConfig, TrainingConfig
from src.data import HateSpeechDataset, load_dataset_files, preprocess_task_b_dataframe, HATE2IDX, IDX2HATE
from src.models import (
    TransplantedKLatentModel,
    PretrainedCrossAttentionLayer,
    HierarchicalTreeHead,
    apply_rotary_pos_emb_single
)
from src.losses import HierarchicalCompoundLoss
from src.metrics import compute_task_b_metrics, evaluate_task_b_by_language
from src.predict import HateSpeechPredictor

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "HateSpeechDataset",
    "load_dataset_files",
    "preprocess_task_b_dataframe",
    "HATE2IDX",
    "IDX2HATE",
    "TransplantedKLatentModel",
    "PretrainedCrossAttentionLayer",
    "HierarchicalTreeHead",
    "apply_rotary_pos_emb_single",
    "HierarchicalCompoundLoss",
    "compute_task_b_metrics",
    "evaluate_task_b_by_language",
    "HateSpeechPredictor"
]
