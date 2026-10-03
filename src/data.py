"""Data loading, cleaning, and dataset classes for Task B (Hate Speech Classification).
"""

import os
import re
import glob
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


# Task B label mapping: 3 classes
HATE2IDX = {
    'no': 0,
    'yes_implicit': 1,
    'yes_explicit': 2
}
IDX2HATE = {v: k for k, v in HATE2IDX.items()}


def clean_multilingual_text(text: Any) -> str:
    """Cleans text without stripping multilingual diacritics or punctuation

    critical for Italian (perché, è), Dutch, and Persian.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_dataset_files(data_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """Loads all StereoQueerEval TSV training files with proper multi-line quoting handling."""
    if not data_paths:
        search_patterns = [
            "./StereoQueerEval_*_training.tsv",
            "./data/StereoQueerEval_*_training.tsv",
            "../StereoQueerEval_*_training.tsv",
            "../LGBT/*_training.tsv",
            "../**/StereoQueerEval_*_training.tsv",
            "./**/StereoQueerEval_*_training.tsv"
        ]
        found = []
        for pat in search_patterns:
            found.extend(glob.glob(pat, recursive=True))
        # Deduplicate paths using resolved absolute paths
        unique_paths = {str(Path(p).resolve()): p for p in found}
        data_paths = sorted(list(unique_paths.values()))

    if not data_paths:
        raise FileNotFoundError(
            "No StereoQueerEval TSV files found. Please ensure files like StereoQueerEval_EN_training.tsv are accessible."
        )

    dfs = []
    for p in data_paths:
        lang_match = re.search(r'_([A-Z]{2})_training\.tsv$', p)
        lang = lang_match.group(1) if lang_match else 'EN'
        df = pd.read_csv(p, sep='\t', quoting=1, encoding='utf-8')
        df['lang'] = lang
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def preprocess_task_b_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses input text triplets and creates the 3-class target column for Task B."""
    df = df.copy()

    # Clean text components
    df['clean_comment'] = df['yt_comment'].apply(clean_multilingual_text)
    df['clean_title'] = df['yt_title'].apply(clean_multilingual_text)
    df['clean_desc'] = df['yt_description'].apply(clean_multilingual_text)

    # Form formatted context string
    # Context format: "Comment: <comment> [SEP] Video: <title> [SEP] Description: <desc>"
    df['formatted_text'] = (
        "Comment: " + df['clean_comment'] +
        " [SEP] Video: " + df['clean_title'] +
        " [SEP] Description: " + df['clean_desc']
    )

    # Map hate speech labels: 0: no, 1: yes_implicit, 2: yes_explicit
    if 'hate_speech' in df.columns:
        mapped = df['hate_speech'].str.strip().str.lower().map(HATE2IDX)
        unmapped = mapped.isna()
        if unmapped.any():
            bad_labels = df.loc[unmapped, 'hate_speech'].unique().tolist()
            print(f"⚠️ Warning: {unmapped.sum()} rows have unknown hate_speech labels: {bad_labels}. Dropping them.")
            df = df[~unmapped].copy()
            mapped = mapped[~unmapped]
        df['label'] = mapped.astype(np.int64)

    return df


def split_by_video_group(
    df: pd.DataFrame,
    val_size: float = 0.15,
    group_col: str = 'yt_title',
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into train/val grouped by video title to avoid data leakage

    since video context is part of the model input.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


class HateSpeechDataset(Dataset):
    """PyTorch Dataset for Task B Hate Speech Classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 384,
        is_training: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        # Pre-tokenize all texts
        self.texts = self.df['formatted_text'].tolist()
        sep = self.tokenizer.sep_token or '</s>'
        encoded = self.tokenizer(
            [t.replace('[SEP]', sep) for t in self.texts],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']

        self.has_labels = 'label' in self.df.columns
        if self.has_labels:
            self.labels = torch.tensor(self.df['label'].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        if self.has_labels:
            item['label'] = self.labels[idx]
        return item
