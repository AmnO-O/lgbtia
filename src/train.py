"""Training pipeline for Task B using Transplanted K-Latent Cross-Attention & Hierarchical Loss.
"""

import os
import time
import argparse
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight

from src.config import ModelConfig, TrainingConfig
from src.data import (
    load_dataset_files,
    preprocess_task_b_dataframe,
    split_by_video_group,
    HateSpeechDataset,
    HATE2IDX,
    IDX2HATE
)
from src.models import TransplantedKLatentModel
from src.losses import HierarchicalCompoundLoss
from src.metrics import compute_task_b_metrics, evaluate_task_b_by_language


def create_optimizer_and_scheduler(
    model: TransplantedKLatentModel,
    train_config: TrainingConfig,
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, Any]:
    """Sets up AdamW optimizer with differential parameter groups."""
    encoder_params = []
    transplant_params = []
    head_and_probes_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            encoder_params.append(param)
        elif 'transplant_layers' in name:
            transplant_params.append(param)
        else:
            head_and_probes_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': train_config.encoder_lr, 'weight_decay': train_config.weight_decay})
    if transplant_params:
        param_groups.append({'params': transplant_params, 'lr': train_config.transplant_lr, 'weight_decay': train_config.weight_decay})
    if head_and_probes_params:
        param_groups.append({'params': head_and_probes_params, 'lr': train_config.head_lr, 'weight_decay': train_config.weight_decay * 0.1})

    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = int(num_training_steps * train_config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate_epoch(
    model: TransplantedKLatentModel,
    val_loader: DataLoader,
    val_df: pd.DataFrame,
    loss_fn: HierarchicalCompoundLoss,
    device: str
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs validation evaluation and computes Macro F1 and per-language breakdowns."""
    model.eval()
    val_loss_sum = 0.0
    all_preds = []
    all_labels = []

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logit_h, logit_f, compound_probs, _ = model(input_ids, attention_mask)
        loss, _ = loss_fn(logit_h, logit_f, labels)
        val_loss_sum += loss.item()

        preds = torch.argmax(compound_probs, dim=-1).cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_loss = val_loss_sum / max(1, len(val_loader))

    metrics = compute_task_b_metrics(all_labels, all_preds)
    metrics['val_loss'] = val_loss

    breakdown_df = evaluate_task_b_by_language(val_df, all_preds)
    return metrics, breakdown_df


def train_task_b_model(
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    df_raw: Optional[pd.DataFrame] = None
) -> Tuple[TransplantedKLatentModel, Dict[str, Any]]:
    """Complete training routine for Task B with K-Latent Transplant Architecture."""
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainingConfig()

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    device = torch.device(train_cfg.device)
    print(f"=== Task B: K-Latent Transplant mmBERT (K={model_cfg.num_query_slots}) on {device} ===")

    # Set all random seeds for reproducibility
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    # 1. Load Data
    if df_raw is None:
        df_raw = load_dataset_files()
    print(f"Loaded {len(df_raw)} records across languages.")

    df_processed = preprocess_task_b_dataframe(df_raw)

    # 2. Group Split by video title to avoid data leakage
    train_df, val_df = split_by_video_group(
        df_processed,
        val_size=train_cfg.val_size,
        group_col=train_cfg.group_col,
        seed=train_cfg.seed
    )
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print("Class distribution (Train):", train_df['label'].value_counts().to_dict())

    # Class weights calculation
    if train_cfg.use_class_weights:
        classes = np.array([0, 1, 2])
        weights = compute_class_weight('balanced', classes=classes, y=train_df['label'].values)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Class Weights: {weights.round(3).tolist()}")
    else:
        class_weights_tensor = None

    # 3. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length=train_cfg.max_length)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length=train_cfg.max_length, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(train_cfg.device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(train_cfg.device == 'cuda')
    )

    # 4. Model & Loss
    model = TransplantedKLatentModel(
        model_name_or_path=model_cfg.model_name_or_path,
        num_query_slots=model_cfg.num_query_slots,
        split_layer_idx=model_cfg.split_layer_idx,
        transplant_layers_count=model_cfg.transplant_layers_count,
        context_mode=model_cfg.context_mode,
        use_prenorm=model_cfg.use_prenorm,
        use_ffn=model_cfg.use_ffn_transplant,
        freeze_encoder=model_cfg.freeze_encoder,
        freeze_transplant=model_cfg.freeze_transplant,
        probe_init_std=model_cfg.probe_init_std,
        dropout=model_cfg.dropout
    ).to(device)

    loss_fn = HierarchicalCompoundLoss(
        class_weights=class_weights_tensor
    ).to(device)

    # Count trainable vs frozen parameters
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: {trainable_p:,} Trainable | {frozen_p:,} Frozen (Strict Isolation Protocol)")

    # 5. Optimizer & Scheduler
    total_steps = (len(train_loader) // train_cfg.gradient_accumulation_steps) * train_cfg.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_cfg, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.fp16 and train_cfg.device == 'cuda')

    best_macro_f1 = -1.0
    patience_counter = 0
    best_metrics = {}

    for epoch in range(train_cfg.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.amp.autocast('cuda', enabled=train_cfg.fp16 and train_cfg.device == 'cuda'):
                logit_h, logit_f, _, _ = model(input_ids, attention_mask)
                loss, _ = loss_fn(logit_h, logit_f, labels)
                loss = loss / train_cfg.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: NaN/Inf loss encountered at step {step}. Skipping step and clearing gradients.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"⚠️ Warning: NaN/Inf gradient norm encountered at step {step}. Skipping optimizer update.")
                    optimizer.zero_grad()
                else:
                    scaler.step(optimizer)
                    scheduler.step()
                scaler.update()
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    optimizer.zero_grad()

            train_loss += loss.item() * train_cfg.gradient_accumulation_steps
            valid_steps += 1

        train_loss = train_loss / max(1, valid_steps)

        # Validation
        val_metrics, breakdown_df = evaluate_epoch(model, val_loader, val_df, loss_fn, device)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1}/{train_cfg.epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"(Acc: {val_metrics['accuracy']:.4f}, F1-No: {val_metrics.get('f1_no', 0):.3f}, "
            f"F1-Implicit: {val_metrics.get('f1_yes_implicit', 0):.3f}, F1-Explicit: {val_metrics.get('f1_yes_explicit', 0):.3f}) | "
            f"Time: {elapsed:.1f}s"
        )

        curr_macro_f1 = val_metrics['macro_f1']
        if curr_macro_f1 > best_macro_f1:
            best_macro_f1 = curr_macro_f1
            best_metrics = val_metrics
            patience_counter = 0

            checkpoint_path = os.path.join(train_cfg.output_dir, "best_task_b_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_cfg,
                'train_config': train_cfg,
                'val_metrics': val_metrics,
                'epoch': epoch + 1
            }, checkpoint_path)
            print(f"✨ New best Task B model saved to {checkpoint_path} (Macro-F1: {best_macro_f1:.4f})")
            print("Language Breakdown:\n", breakdown_df.to_string(index=False))
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break

    return model, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task B with K-Latent Pretrained Weight Borrowing")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmbert-base")
    parser.add_argument("--num_query_slots", type=int, default=8, help="Number of latent query slots K")
    parser.add_argument("--probe_init_std", type=float, default=0.02)
    parser.add_argument("--split_layer_idx", type=int, default=17)
    parser.add_argument("--transplant_layers_count", type=int, default=5)
    parser.add_argument("--context_mode", type=str, choices=["layer_matched", "static_h17"], default="layer_matched")
    parser.add_argument("--use_prenorm", action="store_true", help="Experiment B: Pre-Normalized Borrowing")
    parser.add_argument("--use_ffn", action="store_true", help="Ablation 4: Include transplanted FFN")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    parser.add_argument("--unfreeze_transplant", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--transplant_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs_task_b")
    args = parser.parse_args()

    m_cfg = ModelConfig(
        model_name_or_path=args.model_name,
        num_query_slots=args.num_query_slots,
        probe_init_std=args.probe_init_std,
        split_layer_idx=args.split_layer_idx,
        transplant_layers_count=args.transplant_layers_count,
        context_mode=args.context_mode,
        use_prenorm=args.use_prenorm,
        use_ffn_transplant=args.use_ffn,
        freeze_encoder=not args.unfreeze_encoder,
        freeze_transplant=not args.unfreeze_transplant,
        dropout=args.dropout
    )
    t_cfg = TrainingConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        val_size=args.val_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        encoder_lr=args.encoder_lr,
        transplant_lr=args.transplant_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        use_class_weights=not args.no_class_weights,
        num_workers=args.num_workers
    )

    train_task_b_model(m_cfg, t_cfg)
