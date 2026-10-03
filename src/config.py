"""Configuration for StereoQueerEval Task B (Pretrained Attention Borrowing).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    # Model backbone
    model_name_or_path: str = "jhu-clsp/mmbert-base"
    num_classes: int = 3
    hidden_dim: int = 768
    num_heads: int = 12
    
    # K Query Latents: Configurable bottleneck slots (e.g. K = 4, 8, 16, 32)
    num_query_slots: int = 8
    probe_init_std: float = 0.02
    
    # Layer Split
    split_layer_idx: int = 17            # mmBERT layers 0..16 -> H_17
    transplant_layers_count: int = 5      # mmBERT layers 17..21 (Layers 18 to 22)
    
    # Context Stream Alignment Mode:
    # - 'layer_matched': Layer l receives H_{l-1} (Layer 18 receives H_17, Layer 19 receives H_18, etc.)
    #                    This preserves the exact input distribution pretrained projections were trained on.
    # - 'static_h17': All transplant layers attend to the static representation H_17.
    context_mode: str = "layer_matched"   # 'layer_matched' | 'static_h17'
    
    # Experiment variants from blueprint
    use_prenorm: bool = False             # False = Exp A (Pure Projections), True = Exp B (Pre-LN)
    use_ffn_transplant: bool = False      # False = Projections only, True = Include FFN (Ablation 4)
    freeze_transplant: bool = True        # True = Frozen W_Q, W_K, W_V, W_O (Isolated Hypothesis)
    freeze_encoder: bool = True           # True = Frozen Backbone Layers 1..17
    
    # Hierarchical Head
    dropout: float = 0.1
    classes: List[str] = field(default_factory=lambda: [
        "no", "yes_implicit", "yes_explicit"
    ])


@dataclass
class TrainingConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs_task_b"
    seed: int = 42
    
    # Leakage prevention
    val_size: float = 0.15
    group_col: str = "yt_title"
    
    max_length: int = 384
    
    # Optimization
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    epochs: int = 10
    
    # Learning rates
    encoder_lr: float = 1e-5
    transplant_lr: float = 3e-5
    head_lr: float = 2e-4
    
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Loss & Class Weighting
    use_class_weights: bool = True
    
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 5
