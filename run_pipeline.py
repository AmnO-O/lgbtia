#!/usr/bin/env python3
"""CLI for Task B with K-Latent Attention Transplant Architecture.
"""

import argparse
from src.config import ModelConfig, TrainingConfig
from src.train import train_task_b_model
from src.predict import HateSpeechPredictor


def main():
    parser = argparse.ArgumentParser(description="Task B Pretrained Attention Borrowing Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmbert-base")
    parser.add_argument("--num_query_slots", type=int, default=8, help="Number of latent queries K (e.g. 4, 8, 16)")
    parser.add_argument("--probe_init_std", type=float, default=0.02, help="Initialization standard deviation for query probes")
    parser.add_argument("--split_layer_idx", type=int, default=17, help="Index of backbone layer where context is extracted")
    parser.add_argument("--transplant_layers_count", type=int, default=5, help="Number of upper attention layers to transplant")
    parser.add_argument("--context_mode", type=str, choices=["layer_matched", "static_h17"], default="layer_matched",
                        help="'layer_matched' (layer l receives H_{l-1}) vs 'static_h17' (all layers receive H_17)")
    parser.add_argument("--use_prenorm", action="store_true", help="Experiment B: Pre-Normalized Borrowing")
    parser.add_argument("--use_ffn", action="store_true", help="Ablation 4: Include transplanted FFN")
    parser.add_argument("--unfreeze_encoder", action="store_true", help="Unfreeze backbone context encoder (default: frozen)")
    parser.add_argument("--unfreeze_transplant", action="store_true", help="Unfreeze transplanted projection matrices (default: frozen)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in classification head")
    
    # Training parameters
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
    parser.add_argument("--no_class_weights", action="store_true", help="Disable balanced class weighting in loss")
    parser.add_argument("--output_dir", type=str, default="./outputs_task_b")
    
    # Prediction parameters
    parser.add_argument("--checkpoint", type=str, default="./outputs_task_b/best_task_b_model.pt")
    parser.add_argument("--test_tsv", type=str, default=None)
    parser.add_argument("--submission_tsv", type=str, default="./outputs_task_b/submission_task_b.tsv")

    args = parser.parse_args()

    if args.mode == "train":
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
        print(f"Starting Task B Training (context_mode={args.context_mode}, K={args.num_query_slots})...")
        train_task_b_model(m_cfg, t_cfg)

    elif args.mode == "predict":
        if not args.test_tsv:
            print("Error: Please provide --test_tsv <path_to_tsv> for prediction.")
            return
        predictor = HateSpeechPredictor(checkpoint_path=args.checkpoint)
        predictor.predict_tsv(args.test_tsv, args.submission_tsv)


if __name__ == "__main__":
    main()
