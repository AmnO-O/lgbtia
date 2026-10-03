"""Exact hierarchical 3-class negative log-likelihood for Task B.

Tree:
    root
    ├── no-hate
    └── hate
        ├── implicit
        └── explicit

Given:
    logit_h = hate vs no-hate
    logit_f = implicit vs explicit | hate

The induced class probabilities are:
    P(no)       = sigmoid(-logit_h)
    P(implicit) = sigmoid(logit_h) * sigmoid(logit_f)
    P(explicit) = sigmoid(logit_h) * sigmoid(-logit_f)

Sum constraint:
    P(no) + P(implicit) + P(explicit) == 1.0 identically.
    logsumexp(log_p_no, log_p_implicit, log_p_explicit) == 0.0.

Class Weighting Behavior:
    When class_weights [w_0, w_1, w_2] are provided, this computes weighted leaf-level NLL:
        L = sum_i(w_{y_i} * (-log P_{y_i})) / sum_i(w_{y_i})

    For individual examples:
        L_{no}       = -log sigma(-logit_h)
        L_{implicit} = -log sigma(logit_h) - log sigma(logit_f)
        L_{explicit} = -log sigma(logit_h) - log sigma(-logit_f)

    Thus, class weighting controls how strongly each leaf example influences the root
    node (logit_h) as well as the fine-grained division node (logit_f).
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCompoundLoss(nn.Module):
    """Exact hierarchical 3-class negative log-likelihood with leaf-level class weighting."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(3, dtype=torch.float32))
            self.has_custom_weights = False
        else:
            weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32
            )

            if weights.ndim != 1 or weights.numel() != 3:
                raise ValueError(
                    "class_weights must have shape [3]."
                )

            if torch.any(weights <= 0):
                raise ValueError(
                    "class_weights must be strictly positive."
                )

            self.register_buffer(
                "class_weights",
                weights
            )
            self.has_custom_weights = True

    def forward(
        self,
        logit_h: torch.Tensor,
        logit_f: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logit_h = logit_h.float().reshape(-1)
        logit_f = logit_f.float().reshape(-1)
        targets = targets.long().reshape(-1)

        if logit_h.shape != logit_f.shape:
            raise ValueError(
                "logit_h and logit_f must have identical shape."
            )

        if targets.shape[0] != logit_h.shape[0]:
            raise ValueError(
                "targets and logits must have identical batch size."
            )

        if torch.any((targets < 0) | (targets > 2)):
            raise ValueError(
                "targets must contain only 0, 1, or 2."
            )

        log_p_no = F.logsigmoid(-logit_h)
        log_p_hate = F.logsigmoid(logit_h)

        log_p_implicit = (
            log_p_hate
            + F.logsigmoid(logit_f)
        )

        log_p_explicit = (
            log_p_hate
            + F.logsigmoid(-logit_f)
        )

        log_probs = torch.stack(
            [
                log_p_no,
                log_p_implicit,
                log_p_explicit,
            ],
            dim=-1
        )

        loss = F.nll_loss(
            log_probs,
            targets,
            weight=self.class_weights if self.has_custom_weights else None,
            reduction="mean"
        )

        with torch.no_grad():
            probs = log_probs.exp()

            mean_no = float(-log_p_no[targets == 0].mean()) if (targets == 0).any() else 0.0
            mean_implicit = float(-log_p_implicit[targets == 1].mean()) if (targets == 1).any() else 0.0
            mean_explicit = float(-log_p_explicit[targets == 2].mean()) if (targets == 2).any() else 0.0

            breakdown = {
                "loss_total": float(loss.detach()),
                "mean_nll_no": mean_no,
                "mean_nll_implicit": mean_implicit,
                "mean_nll_explicit": mean_explicit,
                # Backward-compatible aliases
                "loss_no": mean_no,
                "loss_implicit": mean_implicit,
                "loss_explicit": mean_explicit,
                "prob_sum_error": float(
                    (probs.sum(dim=-1) - 1.0).abs().max()
                ),
            }

        return loss, breakdown
