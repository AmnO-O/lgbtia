"""Training pipeline for Task B using Transplanted K-Latent Cross-Attention & Hierarchical Loss.
"""

import os
import time
import math
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
from sklearn.metrics import classification_report

from src.config import ModelConfig, TrainingConfig
from src.data import (
    load_dataset_files,
    preprocess_task_b_dataframe,
    split_by_video_group,
    HateSpeechDataset,
    HATE2IDX,
    IDX2HATE
)
from src.models import (
    TransplantedKLatentModel,
    print_parameter_breakdown,
    verify_parameter_isolation
)
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
    total_val_samples = 0
    all_preds = []
    all_labels = []

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        role_ids = batch.get('role_ids', None)
        if role_ids is not None:
            role_ids = role_ids.to(device)
        labels = batch['label'].to(device)
        bs = labels.size(0)

        logit_h, logit_f, compound_probs, _ = model(input_ids, attention_mask, role_ids=role_ids)
        loss, _ = loss_fn(logit_h, logit_f, labels)
        val_loss_sum += loss.item() * bs
        total_val_samples += bs

        preds = torch.argmax(compound_probs, dim=-1).cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_loss = val_loss_sum / max(1, total_val_samples)

    metrics = compute_task_b_metrics(all_labels, all_preds)
    metrics['val_loss'] = val_loss
    metrics['report'] = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=['no', 'yes_implicit', 'yes_explicit'],
        digits=4,
        zero_division=0
    )

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

    # Set random seeds for reproducibility (distinguishing seeded vs bitwise deterministic)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    if train_cfg.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("🔒 Strict bitwise deterministic training enabled (cuDNN deterministic, benchmark disabled).")
        except Exception as e:
            print(f"⚠️ Warning: Could not enable strict deterministic algorithms ({e}).")
    else:
        # Default seeded reproducibility: fast cuDNN benchmarking allowed
        torch.backends.cudnn.benchmark = True
        print("🌱 Seeded reproducibility enabled (identical data shuffle, weights init; fast non-deterministic GPU kernels).")

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

    # 3. Tokenizer & DataLoaders with deterministic worker initialization
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length=train_cfg.max_length)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length=train_cfg.max_length)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(train_cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(train_cfg.device == 'cuda'),
        worker_init_fn=seed_worker,
        generator=g
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
        dropout=model_cfg.dropout,
        use_context_role_ids=model_cfg.use_context_role_ids,
        num_context_roles=model_cfg.num_context_roles,
        role_init_std=model_cfg.role_init_std
    ).to(device)

    loss_fn = HierarchicalCompoundLoss(
        class_weights=class_weights_tensor
    ).to(device)

    # Explicit Parameter Isolation & Family Breakdown Audit
    print_parameter_breakdown(model)

    # 5. Optimizer & Scheduler
    updates_per_epoch = math.ceil(len(train_loader) / train_cfg.gradient_accumulation_steps)
    total_steps = updates_per_epoch * train_cfg.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_cfg, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.fp16 and train_cfg.device == 'cuda')

    best_macro_f1 = -1.0
    patience_counter = 0
    best_metrics = {}
    best_breakdown_df = pd.DataFrame()
    history = []

    for epoch in range(train_cfg.epochs):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        total_train_samples = 0
        optimizer.zero_grad()

        total_batches = len(train_loader)
        accum_steps = train_cfg.gradient_accumulation_steps

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            role_ids = batch.get('role_ids', None)
            if role_ids is not None:
                role_ids = role_ids.to(device)
            labels = batch['label'].to(device)
            bs = labels.size(0)

            # Determine actual number of microbatches in this accumulation group
            group_start = (step // accum_steps) * accum_steps
            group_end = min(group_start + accum_steps, total_batches)
            current_accum_steps = group_end - group_start

            with torch.amp.autocast('cuda', enabled=train_cfg.fp16 and train_cfg.device == 'cuda'):
                logit_h, logit_f, _, _ = model(input_ids, attention_mask, role_ids=role_ids)
                raw_loss, _ = loss_fn(logit_h, logit_f, labels)
                loss = raw_loss / current_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: NaN/Inf loss encountered at step {step}. Skipping step and clearing gradients.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            is_accum_step = ((step + 1) % accum_steps == 0) or ((step + 1) == total_batches)
            if is_accum_step:
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

            train_loss_sum += raw_loss.item() * bs
            total_train_samples += bs

        train_loss = train_loss_sum / max(1, total_train_samples)

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

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_accuracy': val_metrics['accuracy'],
            'f1_no': val_metrics.get('f1_no', 0.0),
            'f1_implicit': val_metrics.get('f1_yes_implicit', 0.0),
            'f1_explicit': val_metrics.get('f1_yes_explicit', 0.0)
        })

        curr_macro_f1 = val_metrics['macro_f1']
        if curr_macro_f1 > best_macro_f1:
            best_macro_f1 = curr_macro_f1
            best_metrics = val_metrics
            best_breakdown_df = breakdown_df
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

    results = {
        'best_val_macro_f1': best_macro_f1,
        'best_metrics': best_metrics,
        'final_metrics': best_metrics,
        'language_breakdown': best_breakdown_df,
        'history': history,
        **best_metrics
    }
    return model, results


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
    parser.add_argument("--deterministic", action="store_true", help="Enable strict bitwise deterministic training (cuDNN deterministic)")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_context_role_ids", action="store_true", help="Disable context role embeddings")
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
        dropout=args.dropout,
        use_context_role_ids=not args.no_context_role_ids
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
        deterministic=args.deterministic,
        num_workers=args.num_workers
    )

    train_task_b_model(m_cfg, t_cfg)
