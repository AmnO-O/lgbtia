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
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmbert-base")
    parser.add_argument("--num_query_slots", type=int, default=8, help="Number of latent queries K (e.g. 4, 8, 16)")
    parser.add_argument("--context_mode", type=str, choices=["layer_matched", "static_h17"], default="layer_matched",
                        help="'layer_matched' (layer l receives H_{l-1}) vs 'static_h17' (all layers receive H_17)")
    parser.add_argument("--use_prenorm", action="store_true", help="Experiment B: Pre-Normalized Borrowing")
    parser.add_argument("--use_ffn", action="store_true", help="Ablation 4: Include transplanted FFN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--output_dir", type=str, default="./outputs_task_b")
    parser.add_argument("--checkpoint", type=str, default="./outputs_task_b/best_task_b_model.pt")
    parser.add_argument("--test_tsv", type=str, default=None)
    parser.add_argument("--submission_tsv", type=str, default="./outputs_task_b/submission_task_b.tsv")

    args = parser.parse_args()

    if args.mode == "train":
        m_cfg = ModelConfig(
            model_name_or_path=args.model_name,
            num_query_slots=args.num_query_slots,
            context_mode=args.context_mode,
            use_prenorm=args.use_prenorm,
            use_ffn_transplant=args.use_ffn
        )
        t_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            head_lr=args.head_lr,
            max_length=args.max_length,
            output_dir=args.output_dir
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
