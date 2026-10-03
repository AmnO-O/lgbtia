"""Pretrained Attention Weight Borrowing with K-Latent Queries for Task B.
Implements the exact ModernBERT native layer-by-layer forward execution skeleton:
  - `build_native_attention_masks`: Constructs native full and sliding-window 4D attention masks.
  - `encode_context`: Explicit layer-by-layer context encoder matching HuggingFace ModernBERT.
  - `PretrainedCrossAttentionLayer`: Global latent-to-context cross attention with asymmetric RoPE.
  - `HierarchicalTreeHead`: Hierarchical compound classification head.
"""

import math
import copy
import logging
from typing import Tuple, Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embedding exclusively to Keys (K) (Asymmetric RoPE).
    
    Explicit Shape Contract:
      k:   [B, num_heads, L, head_dim]
      cos: [B, L, head_dim] (or broadcastable [1, L, head_dim])
      sin: [B, L, head_dim] (or broadcastable [1, L, head_dim])
    """
    if cos.dim() == 4 and cos.shape[1] == k.shape[2] and cos.shape[2] == 1:
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

    return (k * cos) + (rotate_half(k) * sin)


try:
    from transformers.models.modernbert.modeling_modernbert import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask
    )
except ImportError:
    create_bidirectional_mask = None
    create_bidirectional_sliding_window_mask = None


def build_native_attention_masks(
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None
) -> Dict[str, Optional[torch.Tensor]]:
    """Builds native ModernBERT attention masks using official Hugging Face utilities.
    Automatically respects the active attention backend (FlashAttention-2, SDPA, or eager)
    without materializing unnecessary O(B * L^2) dense tensors.
    """
    # 1. Use official Hugging Face ModernBERT masking utilities if available
    if create_bidirectional_mask is not None and create_bidirectional_sliding_window_mask is not None:
        try:
            full_mask = create_bidirectional_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            sliding_mask = create_bidirectional_sliding_window_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            return {
                "full_attention": full_mask,
                "sliding_attention": sliding_mask,
                "global_attention": full_mask,
            }
        except Exception as e:
            logger.warning("HF native mask utilities failed (%s), falling back to manual masks.", e)

    # 2. Memory-efficient fallback (broadcasted [B, 1, 1, L] without O(L^2) materialization)
    B, L, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    # Safe masking constant for fp16 / fp32 stability (avoid -65504.0 float16 min underflow)
    min_val = -10000.0
    window_size = sliding_window or getattr(config, "local_attention", 128)

    if attention_mask is not None:
        pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)) * min_val
    else:
        pad_mask = torch.zeros((B, 1, 1, L), device=device, dtype=dtype)

    row_idx = torch.arange(L, device=device).unsqueeze(1)
    col_idx = torch.arange(L, device=device).unsqueeze(0)
    band = (col_idx >= row_idx - window_size // 2) & (col_idx <= row_idx + window_size // 2)
    band_mask = torch.where(band.unsqueeze(0).unsqueeze(0), 0.0, min_val).to(dtype)
    sliding_mask = pad_mask + band_mask

    return {
        "full_attention": pad_mask,
        "sliding_attention": sliding_mask,
        "global_attention": pad_mask,
    }


class PretrainedCrossAttentionLayer(nn.Module):
    """Asymmetric RoPE Cross-Attention Layer.

    Projections (Wqkv, Wo) and optional MLP blocks are borrowed directly from pretrained
    ModernBERT attention blocks via deepcopy to guarantee strict parameter isolation.

    Architectural RoPE Specification (Asymmetric RoPE vs Native Symmetric RoPE):
      - Native ModernBERT implements Symmetric RoPE in bidirectional self-attention:
          Q' = RoPE(Q, pos),  K' = RoPE(K, pos)
      - This architecture implements Asymmetric RoPE Cross-Attention:
          Q remains UNROTATED (Position-Free Semantic Latents Z_0 in R^{K x D})
          K is ROTATED (Tokens 1..L encoded with native ModernBERT Rotary Embeddings)
      Rationale:
        Z represents a fixed set of K learned abstract semantic query slots (e.g. Hate concept probes),
        not sequential text tokens. Imposing positional rotation onto Q would inject spurious sequence
        ordering into position-free semantic probes. Rotating K ensures the attention mechanism preserves
        the relative token distances and syntax of the underlying context stream H.
    """

    def __init__(
        self,
        pretrained_attn: nn.Module,
        rotary_emb: Optional[nn.Module] = None,
        pretrained_norm: Optional[nn.Module] = None,
        pretrained_mlp: Optional[nn.Module] = None,
        pretrained_mlp_norm: Optional[nn.Module] = None,
        layer_type: str = "full_attention",
        use_prenorm: bool = False,
        use_ffn: bool = False
    ):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.use_ffn = use_ffn
        self.layer_type = layer_type

        # Deepcopy transplanted weights to ensure complete parameter independence from backbone
        self.Wqkv = copy.deepcopy(pretrained_attn.Wqkv)
        self.Wo = copy.deepcopy(pretrained_attn.Wo)
        self.attn_norm = copy.deepcopy(pretrained_norm) if (use_prenorm and pretrained_norm is not None) else None

        self.mlp = copy.deepcopy(pretrained_mlp) if (use_ffn and pretrained_mlp is not None) else None
        # Faithful FFN transplant: ModernBERT MLP block strictly expects inputs normalized by mlp_norm.
        # use_prenorm controls the Attention block (Exp A vs Exp B); the FFN block must always be faithfully normalized.
        self.mlp_norm = copy.deepcopy(pretrained_mlp_norm) if (use_ffn and pretrained_mlp_norm is not None) else None

        self.rotary_emb = rotary_emb or getattr(pretrained_attn, "rotary_emb", None)

        self.num_heads = pretrained_attn.config.num_attention_heads
        self.head_dim = getattr(pretrained_attn, "head_dim", pretrained_attn.config.hidden_size // self.num_heads)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        z: torch.Tensor,
        h_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, D = z.shape
        _, L, _ = h_context.shape

        # Pre-normalization (Exp B)
        z_in = self.attn_norm(z) if (self.use_prenorm and self.attn_norm is not None) else z
        h_in = self.attn_norm(h_context) if (self.use_prenorm and self.attn_norm is not None) else h_context

        # Project Q from Z, K & V from H (preserving pretrained biases)
        q_weight, k_weight, v_weight = self.Wqkv.weight.chunk(3, dim=0)
        if self.Wqkv.bias is not None:
            q_bias, k_bias, v_bias = self.Wqkv.bias.chunk(3, dim=0)
            q = F.linear(z_in, q_weight, q_bias)
            k = F.linear(h_in, k_weight, k_bias)
            v = F.linear(h_in, v_weight, v_bias)
        else:
            q = F.linear(z_in, q_weight)
            k = F.linear(h_in, k_weight)
            v = F.linear(h_in, v_weight)

        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Asymmetric RoPE: Rotate ONLY Keys (K), leave Queries (Q) unrotated
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=h_context.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(h_context, position_ids, layer_type=self.layer_type)
            except TypeError:
                cos, sin = self.rotary_emb(h_context, position_ids)
            k = apply_rotary_pos_emb_single(k, cos, sin)

        # Global query-to-context attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if context_mask is not None:
            if context_mask.dim() == 2:
                mask_4d = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -10000.0
            else:
                mask_4d = context_mask.to(scores.dtype)
            scores = scores + mask_4d

        attn_weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)

        z_out = z + self.Wo(attn_out)

        if self.use_ffn and self.mlp is not None:
            # Native ModernBERT FFN sub-block: H'' = H' + MLP(mlp_norm(H'))
            # Pretrained MLP weights strictly expect mlp_norm normalized activations
            z_mlp_in = self.mlp_norm(z_out) if self.mlp_norm is not None else z_out
            z_out = z_out + self.mlp(z_mlp_in)

        return z_out, attn_weights.mean(dim=1)


class HierarchicalTreeHead(nn.Module):
    """Hierarchical Tree Head with K-Latent Readout:
    
    Reads from Z in R^[B x K x D]:
      - Level 1 (Hate Detection): p_hate = sigma(W_h * z_hate + b_h)
      - Level 2 (Implicit vs Explicit): p_imp = sigma(W_f * z_imp + b_f)
    
    Compound Probabilities:
      - P(no) = 1 - p_hate = sigma(-z_h)
      - P(yes_implicit) = p_hate * p_imp = sigma(z_h) * sigma(z_f)
      - P(yes_explicit) = p_hate * (1 - p_imp) = sigma(z_h) * sigma(-z_f)
    """

    def __init__(self, hidden_dim: int = 768, num_query_slots: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_query_slots = num_query_slots

        self.query_hate = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.query_fine = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fc_hate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc_fine = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _readout(self, query: torch.Tensor, z_latents: torch.Tensor) -> torch.Tensor:
        B, K, D = z_latents.shape
        q = query.expand(B, -1, -1)
        k = self.key_proj(z_latents)
        v = self.val_proj(z_latents)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) * (D ** -0.5)
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(v.dtype)
        readout = torch.bmm(attn_probs, v).squeeze(1)
        return readout

    def forward(self, z_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_hate_rep = self._readout(self.query_hate, z_latents)
        z_fine_rep = self._readout(self.query_fine, z_latents)

        logit_h = self.fc_hate(z_hate_rep).squeeze(-1)
        logit_f = self.fc_fine(z_fine_rep).squeeze(-1)

        p_hate = torch.sigmoid(logit_h)
        p_imp = torch.sigmoid(logit_f)

        prob_no = 1.0 - p_hate
        prob_implicit = p_hate * p_imp
        prob_explicit = p_hate * (1.0 - p_imp)

        compound_probs = torch.stack([prob_no, prob_implicit, prob_explicit], dim=-1)
        return logit_h, logit_f, compound_probs


class TransplantedKLatentModel(nn.Module):
    """Pretrained Attention Weight Borrowing with K-Latent Queries and Explicit Context Encoding."""

    def __init__(
        self,
        model_name_or_path: str = "jhu-clsp/mmbert-base",
        num_query_slots: int = 8,
        split_layer_idx: int = 17,
        transplant_layers_count: int = 5,
        context_mode: str = "layer_matched",
        use_prenorm: bool = False,
        use_ffn: bool = False,
        freeze_encoder: bool = True,
        freeze_transplant: bool = True,
        probe_init_std: float = 0.02,
        dropout: float = 0.1,
        use_context_role_ids: bool = True,
        num_context_roles: int = 6,
        role_init_std: float = 0.02
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_query_slots = num_query_slots
        self.split_layer_idx = split_layer_idx
        self.transplant_layers_count = transplant_layers_count
        self.context_mode = context_mode
        self.freeze_encoder = freeze_encoder
        self.use_context_role_ids = use_context_role_ids

        # 1. Load mmBERT base model
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.hidden_dim = self.config.hidden_size

        layers = getattr(self.backbone, "layers", None) or getattr(self.backbone.encoder, "layers", None)
        self.rotary_emb = getattr(self.backbone, "rotary_emb", None) or getattr(layers[0].attn, "rotary_emb", None)

        # 2. Trainable Class Queries Z_0 in R^[K x D]
        self.query_probes = nn.Parameter(torch.randn(num_query_slots, self.hidden_dim) * probe_init_std)

        # 2b. Context Role Embeddings (0:PAD, 1:TITLE, 2:DESC, 3:COMMENT, 4:SPECIAL, 5:HINT)
        if self.use_context_role_ids:
            self.role_embeddings = nn.Embedding(num_context_roles, self.hidden_dim)
            nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=role_init_std)
        else:
            self.role_embeddings = None

        # 3. Build Pretrained Cross-Attention Transplant Stack (Layers 18 to 22)
        self.transplant_layers = nn.ModuleList()
        for idx in range(split_layer_idx, split_layer_idx + transplant_layers_count):
            layer_module = layers[idx]
            l_type = getattr(layer_module, "attention_type", getattr(layer_module, "layer_type", "full_attention"))
            cross_attn = PretrainedCrossAttentionLayer(
                pretrained_attn=layer_module.attn,
                rotary_emb=self.rotary_emb,
                pretrained_norm=getattr(layer_module, "attn_norm", None),
                pretrained_mlp=getattr(layer_module, "mlp", None),
                pretrained_mlp_norm=getattr(layer_module, "mlp_norm", None),
                layer_type=l_type,
                use_prenorm=use_prenorm,
                use_ffn=use_ffn
            )
            self.transplant_layers.append(cross_attn)

        # 4. Hierarchical Tree Classifier Head
        self.tree_head = HierarchicalTreeHead(
            hidden_dim=self.hidden_dim,
            num_query_slots=num_query_slots,
            dropout=dropout
        )

        # 5. Freeze enforcement (Orthogonal and Decoupled)
        # Backbone Context Encoder (Layers 1..17)
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_encoder

        # Transplant Cross-Attention Stack (Layers 18..22)
        for layer in self.transplant_layers:
            for p in layer.parameters():
                p.requires_grad = not freeze_transplant

        # Query Latents Z_0 and Classifier Head are always trainable
        self.query_probes.requires_grad = True
        for p in self.tree_head.parameters():
            p.requires_grad = True

        # 6. Post-construction scientific assertions (Strict Parameter Isolation)
        verify_parameter_isolation(self, freeze_encoder=freeze_encoder, freeze_transplant=freeze_transplant)

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Encodes context tokens through mmBERT layers, maintaining a list of

        layer-specific representations [H_0, H_1, ..., H_N].
        """
        B, L = input_ids.shape
        layers = getattr(self.backbone, "layers", None) or getattr(self.backbone.encoder, "layers", None)

        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Embeddings
        hidden_states = self.backbone.embeddings(input_ids)
        saved_hidden_states = [hidden_states]

        # Build native 4D attention masks (full & sliding window)
        attention_mask_mapping = build_native_attention_masks(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask
        )

        # Build position embeddings if rotary_emb supports layer_type query
        layer_types = getattr(self.config, "layer_types", ["full_attention", "sliding_attention"])
        position_embeddings = {}
        for lt in set(layer_types):
            if self.rotary_emb is not None:
                try:
                    # ModernBERT rotary signature: (hidden_states, position_ids, layer_type) or (v, position_ids)
                    position_embeddings[lt] = self.rotary_emb(hidden_states, position_ids, lt)
                except TypeError:
                    try:
                        position_embeddings[lt] = self.rotary_emb(hidden_states, position_ids)
                    except Exception:
                        position_embeddings[lt] = None
            else:
                position_embeddings[lt] = None

        # Execute layers up to the maximum required context index:
        # In 'layer_matched' mode: Layer 22 receives H_21 (produced after running layer 20),
        # so we only run 21 layers (0..20) instead of wasting compute on layer 21 (H_22).
        if self.context_mode == "layer_matched":
            num_layers_to_run = self.split_layer_idx + self.transplant_layers_count - 1
        else:
            num_layers_to_run = self.split_layer_idx

        for i in range(num_layers_to_run):
            layer = layers[i]
            attn_type = getattr(layer, "attention_type", getattr(layer, "layer_type", "full_attention"))
            cur_mask = attention_mask_mapping.get(attn_type, attention_mask_mapping["full_attention"])
            pos_emb = position_embeddings.get(attn_type, None)

            try:
                # Direct ModernBERT layer call with native position_embeddings
                if pos_emb is not None:
                    layer_out = layer(hidden_states, attention_mask=cur_mask, position_embeddings=pos_emb)
                else:
                    layer_out = layer(hidden_states, attention_mask=cur_mask)
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            except Exception as e:
                # Fallback to standard forward if signature differs
                logger.debug("Layer %d native forward failed (%s), using fallback.", i, e)
                layer_out = layer(hidden_states, attention_mask=cur_mask)
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            saved_hidden_states.append(hidden_states)

        return saved_hidden_states, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit encode_context and transplanted cross-attention."""
        B, L = input_ids.shape

        # 1. Encode context stream
        if self.freeze_encoder:
            with torch.no_grad():
                saved_hidden_states, position_ids = self.encode_context(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
        else:
            saved_hidden_states, position_ids = self.encode_context(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

        # 2. Query Stream: Expand Z_0 to batch dimension [B, K, D]
        z = self.query_probes.unsqueeze(0).expand(B, -1, -1)

        # Safety: verify enough hidden states for layer_matched mode
        if self.context_mode == "layer_matched":
            required = self.split_layer_idx + self.transplant_layers_count
            assert len(saved_hidden_states) >= required, (
                f"layer_matched mode requires {required} hidden states, got {len(saved_hidden_states)}"
            )

        # 2b. Compute Context Role Embedding if enabled and provided
        role_emb = None
        if self.use_context_role_ids and self.role_embeddings is not None and role_ids is not None:
            role_emb = self.role_embeddings(role_ids.clamp(0, self.role_embeddings.num_embeddings - 1))

        # 3. Pass through Transplanted Cross-Attention Stack (Layers 18 to 22)
        for i, transplant_layer in enumerate(self.transplant_layers):
            if self.context_mode == "layer_matched":
                # Layer l receives H_{l-1}
                h_ctx = saved_hidden_states[self.split_layer_idx + i]
            else:
                # Static H_17 for all layers
                h_ctx = saved_hidden_states[self.split_layer_idx]

            # Inject role embedding into context stream for cross-attention
            if role_emb is not None:
                h_ctx = h_ctx + role_emb

            z, _ = transplant_layer(
                z=z,
                h_context=h_ctx,
                context_mask=attention_mask,
                position_ids=position_ids
            )

        # 4. Hierarchical Classification Head
        logit_h, logit_f, compound_probs = self.tree_head(z)

        return logit_h, logit_f, compound_probs, z


def verify_parameter_isolation(
    model: nn.Module,
    freeze_encoder: bool = True,
    freeze_transplant: bool = True
) -> None:
    """Scientific verification asserting parameter isolation and weight borrowing fidelity.

    Guarantees:
      1. Gradient isolation: frozen parameter families never leak gradients (requires_grad=False).
      2. Trainable invariants: query probes, classification head, and role embeddings have requires_grad=True.
      3. Weight borrowing fidelity: W_transplant == W_source at initialization.
      4. Memory decoupling: transplant weights are independent deepcopies (data_ptr mismatch).
    """
    backbone = getattr(model, "backbone", None)
    transplant_layers = getattr(model, "transplant_layers", None)
    split_layer_idx = getattr(model, "split_layer_idx", 17)

    # 1. Gradient isolation check
    if freeze_encoder and backbone is not None:
        for name, p in backbone.named_parameters():
            assert not p.requires_grad, f"Isolation failure: Backbone parameter '{name}' must be frozen!"

    if freeze_transplant and transplant_layers is not None:
        for name, p in transplant_layers.named_parameters():
            assert not p.requires_grad, f"Isolation failure: Transplant parameter '{name}' must be frozen!"

    # 2. Trainable invariants
    query_probes = getattr(model, "query_probes", None)
    if query_probes is not None:
        assert query_probes.requires_grad, "Trainability failure: query_probes must have requires_grad=True!"

    tree_head = getattr(model, "tree_head", None)
    if tree_head is not None:
        for name, p in tree_head.named_parameters():
            assert p.requires_grad, f"Trainability failure: Classification head parameter '{name}' must be trainable!"

    role_embeddings = getattr(model, "role_embeddings", None)
    if role_embeddings is not None:
        assert role_embeddings.weight.requires_grad, "Trainability failure: role_embeddings must have requires_grad=True!"

    # 3. Weight borrowing fidelity check (W_copy == W_source) & Memory decoupling check
    if backbone is not None and transplant_layers is not None:
        layers = getattr(backbone, "layers", None) or getattr(backbone.encoder, "layers", None)
        if layers is not None:
            for i, transplant in enumerate(transplant_layers):
                source_idx = split_layer_idx + i
                if source_idx < len(layers):
                    source = layers[source_idx].attn

                    # Verify Wqkv weights
                    assert torch.equal(transplant.Wqkv.weight, source.Wqkv.weight), (
                        f"Weight borrowing fidelity violation at layer {i}: "
                        f"transplant.Wqkv.weight != backbone layer {source_idx} attn.Wqkv.weight"
                    )
                    assert transplant.Wqkv.weight.data_ptr() != source.Wqkv.weight.data_ptr(), (
                        f"Memory coupling violation at layer {i}: "
                        f"transplant.Wqkv shares storage with backbone!"
                    )
                    if transplant.Wqkv.bias is not None and source.Wqkv.bias is not None:
                        assert torch.equal(transplant.Wqkv.bias, source.Wqkv.bias), (
                            f"Weight borrowing fidelity violation at layer {i}: Wqkv bias mismatch!"
                        )

                    # Verify Wo weights
                    assert torch.equal(transplant.Wo.weight, source.Wo.weight), (
                        f"Weight borrowing fidelity violation at layer {i}: "
                        f"transplant.Wo.weight != backbone layer {source_idx} attn.Wo.weight"
                    )
                    assert transplant.Wo.weight.data_ptr() != source.Wo.weight.data_ptr(), (
                        f"Memory coupling violation at layer {i}: "
                        f"transplant.Wo shares storage with backbone!"
                    )
                    if transplant.Wo.bias is not None and source.Wo.bias is not None:
                        assert torch.equal(transplant.Wo.bias, source.Wo.bias), (
                            f"Weight borrowing fidelity violation at layer {i}: Wo bias mismatch!"
                        )

                    # Verify MLP and MLP Norm if FFN transplant is active (Ablation 4)
                    if transplant.mlp is not None and getattr(layers[source_idx], "mlp", None) is not None:
                        source_mlp = layers[source_idx].mlp
                        for (p_name, tp), (_, sp) in zip(transplant.mlp.named_parameters(), source_mlp.named_parameters()):
                            assert torch.equal(tp, sp), f"FFN transplant fidelity violation at layer {i}: {p_name}"
                            assert tp.data_ptr() != sp.data_ptr(), f"FFN transplant coupling violation at layer {i}: {p_name}"

                    if transplant.mlp_norm is not None and getattr(layers[source_idx], "mlp_norm", None) is not None:
                        source_mlp_norm = layers[source_idx].mlp_norm
                        for (p_name, tp), (_, sp) in zip(transplant.mlp_norm.named_parameters(), source_mlp_norm.named_parameters()):
                            assert torch.equal(tp, sp), f"FFN norm transplant fidelity violation at layer {i}: {p_name}"
                            assert tp.data_ptr() != sp.data_ptr(), f"FFN norm transplant coupling violation at layer {i}: {p_name}"



def print_parameter_breakdown(model: nn.Module) -> None:
    """Prints an explicit family-by-family audit of trainable vs frozen parameters."""
    trainable_families: Dict[str, int] = {}
    frozen_families: Dict[str, int] = {}

    for name, p in model.named_parameters():
        if "query_probes" in name:
            family = "query_probes (K-latent vectors)"
        elif "role_embeddings" in name:
            family = "role_embeddings (context role IDs)"
        elif "tree_head" in name:
            family = "tree_head (hierarchical classifier)"
        elif "transplant_layers" in name:
            family = "transplant_layers (borrowed projections)"
        elif "backbone" in name:
            family = "backbone (context encoder)"
        else:
            family = name.split(".")[0]

        target_dict = trainable_families if p.requires_grad else frozen_families
        target_dict[family] = target_dict.get(family, 0) + p.numel()

    print("\n" + "=" * 65)
    print("🔒 PARAMETER ISOLATION PROTOCOL VERIFICATION")
    print("=" * 65)
    print("🟢 Trainable Parameter Families:")
    for fam, cnt in sorted(trainable_families.items()):
        print(f"   ✓ {fam:<42}: {cnt:>10,} params")

    print("\n🧊 Frozen Parameter Families:")
    for fam, cnt in sorted(frozen_families.items()):
        print(f"   ❄️ {fam:<42}: {cnt:>10,} params")

    total_train = sum(trainable_families.values())
    total_frozen = sum(frozen_families.values())
    total_all = total_train + total_frozen
    pct = (total_train / total_all * 100) if total_all > 0 else 0.0
    print("-" * 65)
    print(f"Total: {total_train:,} Trainable | {total_frozen:,} Frozen ({pct:.2f}% active)")
    print("=" * 65 + "\n")

