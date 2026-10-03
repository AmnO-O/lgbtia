"""Unit test verifying layer truncation and parameter isolation protocols."""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from src.models import TransplantedKLatentModel, verify_parameter_isolation, print_parameter_breakdown


def test_layer_execution_truncation_static_h17():
    """Verifies that under context_mode='static_h17', the backbone strictly executes

    layers 0..16 (producing H_17) and that layers 17..21 are NEVER executed by the backbone.
    """
    print("\n--- TEST: Backbone Truncation at Layer 17 (static_h17 mode) ---")
    model = TransplantedKLatentModel(
        model_name_or_path="jhu-clsp/mmbert-base",
        num_query_slots=4,
        split_layer_idx=17,
        transplant_layers_count=5,
        context_mode="static_h17",
        freeze_encoder=True,
        freeze_transplant=True
    )
    model.eval()

    backbone_layers = getattr(model.backbone, "layers", None) or getattr(model.backbone.encoder, "layers", None)
    total_backbone_layers = len(backbone_layers)
    layer_call_counts = [0] * total_backbone_layers

    # Register forward hooks on all backbone layers
    def make_hook(idx):
        def hook(module, inp, out):
            layer_call_counts[idx] += 1
        return hook

    hooks = []
    for idx, layer in enumerate(backbone_layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Run dummy forward pass
    dummy_input_ids = torch.randint(0, 1000, (2, 32))
    dummy_mask = torch.ones((2, 32), dtype=torch.long)

    with torch.no_grad():
        logit_h, logit_f, probs, z = model(dummy_input_ids, dummy_mask)

    for h in hooks:
        h.remove()

    print(f"Total backbone layers: {total_backbone_layers}")
    print(f"Layer call counts (0 to 16): {layer_call_counts[:17]}")
    print(f"Layer call counts (17 to 21): {layer_call_counts[17:22]}")

    # Assertions
    # 1. Layers 0 to 16 must be executed exactly once
    for idx in range(17):
        assert layer_call_counts[idx] == 1, f"Layer {idx} should have been called once, got {layer_call_counts[idx]}"

    # 2. Layers 17 to total_backbone_layers must NEVER be called by the backbone
    for idx in range(17, total_backbone_layers):
        assert layer_call_counts[idx] == 0, (
            f"VIOLATION: Backbone layer {idx} was executed by the backbone! "
            f"Call count = {layer_call_counts[idx]}. Expected 0."
        )

    print("✅ TEST PASSED: Backbone strictly stopped at Layer 17 (0..16 executed, 17+ never called by backbone).")


def test_parameter_isolation():
    """Verifies that all parameter families are strictly partitioned and verified."""
    print("\n--- TEST: Parameter Isolation Assertions ---")
    model = TransplantedKLatentModel(
        model_name_or_path="jhu-clsp/mmbert-base",
        num_query_slots=8,
        split_layer_idx=17,
        transplant_layers_count=5,
        freeze_encoder=True,
        freeze_transplant=True,
        use_context_role_ids=True
    )

    # Verification function should pass without exception
    verify_parameter_isolation(model, freeze_encoder=True, freeze_transplant=True)
    print("✅ Post-construction assertions verified successfully.")

    # Audit breakdown display
    print_parameter_breakdown(model)


if __name__ == "__main__":
    test_layer_execution_truncation_static_h17()
    test_parameter_isolation()
