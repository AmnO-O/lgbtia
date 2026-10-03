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

# Canonical Context Role IDs:
# Exactly aligned with model embedding expectations:
# 0: PAD (padding tokens)
# 1: TITLE (video title context)
# 2: DESC (video description context)
# 3: COMMENT (primary hate speech target)
# 4: SPECIAL (BOS, CLS, SEP, EOS delimiter tokens)
# 5: HINT (reserved for optional prompt hints)
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
ROLE_SPECIAL = 4
ROLE_HINT = 5
NUM_ROLES = 6


# Valid training languages in StereoQueerEval Task B
VALID_TRAINING_LANGUAGES = {"EN", "IT", "NL"}


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
        if not lang_match:
            raise ValueError(
                f"Could not infer language from filename: '{p}'. "
                f"Expected pattern '*_<LANG>_training.tsv' (e.g. StereoQueerEval_EN_training.tsv)."
            )
        lang = lang_match.group(1)
        if lang not in VALID_TRAINING_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang}' extracted from '{p}'. "
                f"Expected one of the released training languages: {sorted(VALID_TRAINING_LANGUAGES)}."
            )
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
    """
    Task B dataset with explicit role construction.

    Output:
        input_ids
        attention_mask
        role_ids
        label (when available)

    Role IDs:
        0 = PAD
        1 = TITLE
        2 = DESCRIPTION
        3 = COMMENT
        4 = SPECIAL
        5 = HINT
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 384,
        comment_max_length: int = 192,
        title_max_length: int = 64,
        desc_max_length: int = 124,
        **kwargs
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ---------------------------------------------------------
        # Token IDs (Never use 'or' as token ID 0 is a valid ID)
        # ---------------------------------------------------------
        pad_id = getattr(tokenizer, "pad_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)

        if sep_id is None:
            sep_id = getattr(tokenizer, "eos_token_id", None)

        cls_id = getattr(tokenizer, "cls_token_id", None)
        if cls_id is None:
            cls_id = getattr(tokenizer, "bos_token_id", None)

        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id.")

        if sep_id is None:
            raise ValueError(
                "Tokenizer has neither sep_token_id nor eos_token_id."
            )

        # ---------------------------------------------------------
        # Tokenize fields independently.
        # No special tokens here; we add them explicitly.
        # ---------------------------------------------------------
        comments = self.df["clean_comment"].tolist() if "clean_comment" in self.df.columns else self.df["yt_comment"].apply(clean_multilingual_text).tolist()
        titles = self.df["clean_title"].tolist() if "clean_title" in self.df.columns else self.df["yt_title"].apply(clean_multilingual_text).tolist()
        descs = self.df["clean_desc"].tolist() if "clean_desc" in self.df.columns else self.df["yt_description"].apply(clean_multilingual_text).tolist()

        comment_enc = tokenizer(
            comments,
            add_special_tokens=False,
            truncation=True,
            max_length=comment_max_length,
            padding=False,
        )

        title_enc = tokenizer(
            titles,
            add_special_tokens=False,
            truncation=True,
            max_length=title_max_length,
            padding=False,
        )

        desc_enc = tokenizer(
            descs,
            add_special_tokens=False,
            truncation=True,
            max_length=desc_max_length,
            padding=False,
        )

        input_ids_list = []
        role_ids_list = []
        attention_masks_list = []

        for i in range(len(self.df)):
            ids = []
            roles = []

            # -----------------------------------------------------
            # CLS
            # -----------------------------------------------------
            if cls_id is not None:
                ids.append(cls_id)
                roles.append(ROLE_SPECIAL)

            # -----------------------------------------------------
            # COMMENT
            # -----------------------------------------------------
            comment_ids = comment_enc["input_ids"][i]
            ids.extend(comment_ids)
            roles.extend([ROLE_COMMENT] * len(comment_ids))

            # -----------------------------------------------------
            # SEP
            # -----------------------------------------------------
            ids.append(sep_id)
            roles.append(ROLE_SPECIAL)

            # -----------------------------------------------------
            # TITLE
            # -----------------------------------------------------
            title_ids = title_enc["input_ids"][i]
            ids.extend(title_ids)
            roles.extend([ROLE_TITLE] * len(title_ids))

            # -----------------------------------------------------
            # SEP
            # -----------------------------------------------------
            ids.append(sep_id)
            roles.append(ROLE_SPECIAL)

            # -----------------------------------------------------
            # DESCRIPTION
            # -----------------------------------------------------
            desc_ids = desc_enc["input_ids"][i]
            ids.extend(desc_ids)
            roles.extend([ROLE_DESC] * len(desc_ids))

            # -----------------------------------------------------
            # Final SEP
            # -----------------------------------------------------
            ids.append(sep_id)
            roles.append(ROLE_SPECIAL)

            # -----------------------------------------------------
            # Truncate
            # -----------------------------------------------------
            ids = ids[:max_length]
            roles = roles[:max_length]

            # -----------------------------------------------------
            # Attention mask before padding
            # -----------------------------------------------------
            attention = [1] * len(ids)

            # -----------------------------------------------------
            # Padding
            # -----------------------------------------------------
            padding_length = max_length - len(ids)

            if padding_length > 0:
                ids.extend([pad_id] * padding_length)
                roles.extend([ROLE_PAD] * padding_length)
                attention.extend([0] * padding_length)

            input_ids_list.append(ids)
            role_ids_list.append(roles)
            attention_masks_list.append(attention)

        # ---------------------------------------------------------
        # Tensors
        # ---------------------------------------------------------
        self.input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        self.role_ids = torch.tensor(role_ids_list, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_masks_list, dtype=torch.long)

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------
        self.has_labels = "label" in self.df.columns
        if self.has_labels:
            self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "role_ids": self.role_ids[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item
