"""Inference pipeline for Task B using the K-Latent Transplant Architecture.
"""

import os
import argparse
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import ModelConfig
from src.data import clean_multilingual_text, IDX2HATE
from src.models import TransplantedKLatentModel


class HateSpeechPredictor:
    """Predictor for Task B with K-Latent Borrowed Attention Stack."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint from: {checkpoint_path} on {self.device}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_cfg: ModelConfig = checkpoint.get('model_config', ModelConfig())

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name_or_path)

        self.model = TransplantedKLatentModel(
            model_name_or_path=self.model_cfg.model_name_or_path,
            num_query_slots=self.model_cfg.num_query_slots,
            split_layer_idx=self.model_cfg.split_layer_idx,
            transplant_layers_count=self.model_cfg.transplant_layers_count,
            context_mode=self.model_cfg.context_mode,
            use_prenorm=self.model_cfg.use_prenorm,
            use_ffn=self.model_cfg.use_ffn_transplant,
            freeze_encoder=False,
            freeze_transplant=False,
            dropout=0.0
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def format_input(self, comment: str, title: str = "", description: str = "") -> str:
        c = clean_multilingual_text(comment)
        t = clean_multilingual_text(title)
        d = clean_multilingual_text(description)
        return f"Comment: {c} [SEP] Video: {t} [SEP] Description: {d}"

    @torch.no_grad()
    def predict_batch(
        self,
        comments: List[str],
        titles: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
        batch_size: int = 32,
        max_length: int = 384
    ) -> pd.DataFrame:
        titles = titles or [""] * len(comments)
        descriptions = descriptions or [""] * len(comments)

        formatted_texts = [
            self.format_input(c, t, d)
            for c, t, d in zip(comments, titles, descriptions)
        ]

        all_preds = []
        all_probs = []

        sep = self.tokenizer.sep_token or '</s>'
        for i in range(0, len(formatted_texts), batch_size):
            chunk = formatted_texts[i:i + batch_size]
            encoded = self.tokenizer(
                [t.replace('[SEP]', sep) for t in chunk],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            _, _, compound_probs, _ = self.model(encoded['input_ids'], encoded['attention_mask'])
            probs = compound_probs.cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_probs.extend(probs)

        results = []
        for idx in range(len(comments)):
            pred_idx = int(all_preds[idx])
            pred_label = IDX2HATE[pred_idx]
            p = all_probs[idx]

            results.append({
                'yt_comment': comments[idx],
                'hate_speech_pred': pred_label,
                'confidence': float(p[pred_idx]),
                'prob_no': float(p[0]),
                'prob_yes_implicit': float(p[1]),
                'prob_yes_explicit': float(p[2]),
            })

        return pd.DataFrame(results)

    def predict_tsv(
        self,
        tsv_path: str,
        output_tsv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Runs batch prediction on a test TSV file."""
        print(f"Reading test data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t', quoting=1, encoding='utf-8')

        preds_df = self.predict_batch(
            comments=df['yt_comment'].tolist(),
            titles=df['yt_title'].tolist() if 'yt_title' in df.columns else None,
            descriptions=df['yt_description'].tolist() if 'yt_description' in df.columns else None
        )

        out_df = df.copy()
        out_df['hate_speech'] = preds_df['hate_speech_pred']

        if output_tsv_path:
            out_df.to_csv(output_tsv_path, sep='\t', index=False)
            print(f"Predictions written to {output_tsv_path}")

        return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with K-Latent Transplant Architecture")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tsv", type=str, help="Path to TSV file")
    parser.add_argument("--output", type=str, default="task_b_submission.tsv")
    args = parser.parse_args()

    predictor = HateSpeechPredictor(checkpoint_path=args.checkpoint)
    if args.tsv:
        predictor.predict_tsv(args.tsv, args.output)
    else:
        sample = predictor.predict_batch(
            comments=["These people are not welcome here."],
            titles=["Amsterdam News"],
            descriptions=["Pride week report."]
        )
        print("Sample Output:\n", sample.to_dict(orient='records'))
