"""Compound Log-Sigmoid Hierarchical Tree Loss for Task B.
Implements the exact joint NLL from the blueprint:
  - y = 0 ('no')           : -logsigmoid(-z_h)
  - y = 1 ('yes_implicit') : -logsigmoid(z_h) - logsigmoid(z_f)
  - y = 2 ('yes_explicit') : -logsigmoid(z_h) - logsigmoid(-z_f)
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCompoundLoss(nn.Module):
    """Numerically stable compound log-sigmoid loss for 2-level decision tree."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        level1_weight: float = 1.0,
        level2_weight: float = 1.0
    ):
        super().__init__()
        self.class_weights = class_weights
        self.level1_weight = level1_weight
        self.level2_weight = level2_weight

    def forward(
        self,
        logit_h: torch.Tensor,
        logit_f: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
          - logit_h: Level 1 logit (z_h) in R^[B] (Hate vs Non-Hate)
          - logit_f: Level 2 logit (z_f) in R^[B] (Implicit vs Explicit)
          - targets: Integer tensor in R^[B] with values in {0, 1, 2}
        """
        # Level 1 log-sigmoids
        log_p_hate = F.logsigmoid(logit_h)
        log_p_nohate = F.logsigmoid(-logit_h)

        # Level 2 log-sigmoids
        log_p_imp = F.logsigmoid(logit_f)
        log_p_exp = F.logsigmoid(-logit_f)

        # Compound negative log-likelihoods per class
        loss_no = -log_p_nohate
        loss_implicit = - (self.level1_weight * log_p_hate + self.level2_weight * log_p_imp)
        loss_explicit = - (self.level1_weight * log_p_hate + self.level2_weight * log_p_exp)

        # Select per sample
        losses = torch.zeros_like(logit_h)
        mask_0 = (targets == 0)
        mask_1 = (targets == 1)
        mask_2 = (targets == 2)

        losses[mask_0] = loss_no[mask_0]
        losses[mask_1] = loss_implicit[mask_1]
        losses[mask_2] = loss_explicit[mask_2]

        # Apply class weights if provided
        if self.class_weights is not None:
            if self.class_weights.device != targets.device:
                self.class_weights = self.class_weights.to(targets.device)
            sample_weights = self.class_weights.gather(0, targets)
            losses = losses * sample_weights

        total_loss = losses.mean()

        breakdown = {
            'loss_total': total_loss.item(),
            'loss_no': loss_no[mask_0].mean().item() if mask_0.any() else 0.0,
            'loss_implicit': loss_implicit[mask_1].mean().item() if mask_1.any() else 0.0,
            'loss_explicit': loss_explicit[mask_2].mean().item() if mask_2.any() else 0.0,
        }

        return total_loss, breakdown
