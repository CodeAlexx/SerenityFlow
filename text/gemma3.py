"""Gemma 3 12B text encoder for LTX-2 inference.

Pipeline:
1. Tokenize text with Gemma 3 tokenizer (SentencePiece)
2. Forward through all 48 transformer layers (collect ALL hidden states)
3. Stack 49 hidden states (embed + 48 layers) per token -> 49*3840 = 188160
4. Project to 3840 via text_embedding_projection
5. Normalize: 8.0 * (x - mean) / (max - min + eps)
6. Pass through video_embeddings_connector (2 transformer blocks with RoPE + registers)
7. Output: (B, seq_len, 4096) for video stream of DiT

This module handles steps 1-6 and outputs the 3840-dim representation.
The video_embeddings_connector is loaded as part of this encoder since it's
stored under the main checkpoint, not the Gemma weights.
"""
from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["Gemma3Encoder"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemma 3 12B Config
# ---------------------------------------------------------------------------

@dataclass
class Gemma3Config:
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta_global: float = 1000000.0
    rope_theta_local: float = 10000.0
    head_dim: int = 256
    # Sliding window: layers with local attention use sliding_window_size
    sliding_window_pattern: list = None  # [1024, 1024, 1024, 1024, 1024, False] repeating

    def __post_init__(self):
        if self.sliding_window_pattern is None:
            self.sliding_window_pattern = [1024, 1024, 1024, 1024, 1024, False]


# ---------------------------------------------------------------------------
# RMSNorm (with add variant for Gemma 3)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        self.add = add

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * norm
        w = self.weight.float()
        if self.add:
            x = x * (1.0 + w)
        else:
            x = x * w
        return x.to(dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    """Build cos/sin cache for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor x of shape (B, H, T, D)."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Gemma 3 Attention
# ---------------------------------------------------------------------------

class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK norms (Gemma 3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=False)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=False)

        # Determine if this layer uses sliding window (local) attention
        pattern = config.sliding_window_pattern
        pattern_idx = layer_idx % len(pattern)
        self.is_global = pattern[pattern_idx] is False
        self.sliding_window = None if self.is_global else pattern[pattern_idx]

        # RoPE theta differs for global vs local
        self.rope_theta = config.rope_theta_global if self.is_global else config.rope_theta_local

    def forward(self, x: torch.Tensor, cos_global: torch.Tensor, sin_global: torch.Tensor,
                cos_local: torch.Tensor, sin_local: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE with appropriate theta
        cos = cos_global if self.is_global else cos_local
        sin = sin_global if self.is_global else sin_local
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Gemma 3 MLP
# ---------------------------------------------------------------------------

class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ---------------------------------------------------------------------------
# Gemma 3 Transformer Layer
# ---------------------------------------------------------------------------

class Gemma3Layer(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos_global, sin_global, cos_local, sin_local, attention_mask=None):
        # Pre-norm self-attention
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos_global, sin_global, cos_local, sin_local, attention_mask)
        h = self.post_attention_layernorm(h)
        x = x + h

        # Pre-norm FFN
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Gemma 3 Model
# ---------------------------------------------------------------------------

class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma3Layer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass returning ALL hidden states (embed + each layer output).

        Returns:
            all_hidden_states: list of (B, T, 3840) tensors, length = num_layers + 1
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        # Build RoPE caches
        cos_global, sin_global = build_rope_cache(T, self.config.head_dim, self.config.rope_theta_global, device, dtype)
        cos_local, sin_local = build_rope_cache(T, self.config.head_dim, self.config.rope_theta_local, device, dtype)

        # Causal mask
        causal_mask = None  # SDPA handles causal internally, but we need it for padding
        if attention_mask is not None:
            # Convert (B, T) padding mask to (B, 1, T, T) attention mask
            expanded = attention_mask[:, None, None, :].expand(B, 1, T, T)
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            causal_mask = expanded.clone()
            causal_mask.masked_fill_(causal[None, None], 0)
            causal_mask = causal_mask.to(dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min

        x = self.embed_tokens(input_ids)
        # Gemma 3 normalizes embeddings
        x = x * (self.config.hidden_size ** 0.5)

        all_hidden_states = [x]

        for layer in self.layers:
            x = layer(x, cos_global, sin_global, cos_local, sin_local, causal_mask)
            all_hidden_states.append(x)

        # Final norm on last hidden state
        all_hidden_states[-1] = self.norm(all_hidden_states[-1])

        return all_hidden_states


# ---------------------------------------------------------------------------
# Embeddings1DConnector (transforms Gemma output for DiT)
# ---------------------------------------------------------------------------

class EmbeddingsConnector(nn.Module):
    """1D transformer connector with learnable registers and split RoPE.

    Takes projected Gemma output (B, T, 3840) and processes through
    2 self-attention + FFN blocks with positional encoding.
    Appends learnable register tokens to expand context.
    """

    def __init__(self, inner_dim: int = 3840, num_heads: int = 30, head_dim: int = 128,
                 num_layers: int = 2, num_registers: int = 128,
                 max_pos: list[int] | int | None = None, theta: float = 10000.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_registers = num_registers
        # Reference default: [1], not 4096
        if max_pos is None:
            self.max_pos = [1]
        elif isinstance(max_pos, int):
            self.max_pos = [max_pos]
        else:
            self.max_pos = max_pos
        self.theta = theta

        if num_registers > 0:
            self.learnable_registers = nn.Parameter(torch.randn(num_registers, inner_dim) * 2.0 - 1.0)

        from models.ltx2_dit import (
            CrossAttention as DitCrossAttention,
            FeedForward as DitFeedForward,
            rms_norm,
        )
        self._rms_norm = rms_norm

        self.transformer_1d_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Module()
            block.attn1 = DitCrossAttention(inner_dim, heads=num_heads, dim_head=head_dim)
            block.ff = DitFeedForward(inner_dim, inner_dim)
            self.transformer_1d_blocks.append(block)

    def _precompute_freqs(self, seq_len, device, dtype):
        """Compute interleaved RoPE for 1D sequence."""
        from models.ltx2_dit import generate_freq_grid_np, interleaved_freqs_cis

        indices = generate_freq_grid_np(self.theta, 1, self.inner_dim).to(device)
        # indices shape: (inner_dim // 2,) = (1920,)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frac = positions / self.max_pos[0]  # (seq_len,)

        # (seq_len, 1) * (1920,) broadcast → (seq_len, 1920)
        freqs = indices * (frac.unsqueeze(-1) * 2 - 1)
        freqs = freqs.unsqueeze(0)  # (1, seq_len, 1920)

        # interleaved: pad_size = inner_dim % (2 * n_pos_dims), n_pos_dims=1
        pad_size = self.inner_dim % 2  # 0 for dim=3840
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, pad_size)
        return cos_freq.to(dtype), sin_freq.to(dtype), False

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Process hidden states through connector blocks.

        Args:
            hidden_states: (B, T, 3840) projected Gemma output
            attention_mask: Optional (B, 1, 1, T) float attention mask
                where large negative values indicate padded positions.

        Returns:
            tuple of (hidden_states, attention_mask) where T is unchanged
        """
        if self.num_registers > 0 and attention_mask is not None:
            # Compact valid tokens to the front, fill remaining with tiled registers
            # (matches WanGP _replace_padded_with_learnable_registers)
            seq_len = hidden_states.shape[1]
            assert seq_len % self.num_registers == 0, (
                f"Sequence length {seq_len} must be divisible by num_registers {self.num_registers}"
            )

            num_dups = seq_len // self.num_registers
            learnable_registers = torch.tile(
                self.learnable_registers.to(hidden_states),
                (num_dups, 1),
            )

            # Build binary mask: 1 where valid, 0 where padded
            if attention_mask.ndim == 4:
                # (B, 1, 1, T) -> (B, T, 1)
                attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
            else:
                attention_mask_binary = (attention_mask.unsqueeze(-1) >= -9000.0).int()

            # Extract valid (non-padded) tokens and left-justify them
            non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze(-1).squeeze(0).bool(), :]
            non_zero_nums = non_zero_hidden_states.shape[1]
            pad_length = seq_len - non_zero_nums
            adjusted_hidden_states = F.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)

            # Flip mask: valid tokens are now at the front, registers fill the back
            flipped_mask = torch.flip(attention_mask_binary, dims=[1])
            hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * learnable_registers

            # Clear attention mask -- all positions now attend
            attention_mask = torch.full_like(attention_mask, 0.0)
        elif self.num_registers > 0:
            # No mask -- original append behavior as fallback
            num_dups = math.ceil(max(1024, hidden_states.shape[1]) / self.num_registers)
            regs = torch.tile(
                self.learnable_registers.to(hidden_states),
                (num_dups, 1),
            )
            regs = regs[hidden_states.shape[1]:].unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)
            hidden_states = torch.cat([hidden_states, regs], dim=1)

        # Compute RoPE
        pe = self._precompute_freqs(hidden_states.shape[1], hidden_states.device, hidden_states.dtype)

        for block in self.transformer_1d_blocks:
            # Self-attention with RoPE
            norm_h = self._rms_norm(hidden_states)
            attn_out = block.attn1(norm_h, pe=pe)
            hidden_states = attn_out + hidden_states
            # FFN
            norm_h = self._rms_norm(hidden_states)
            ff_out = block.ff(norm_h)
            hidden_states = ff_out + hidden_states

        hidden_states = self._rms_norm(hidden_states)
        return hidden_states, attention_mask


# ---------------------------------------------------------------------------
# Gemma3Encoder — full text encoding pipeline
# ---------------------------------------------------------------------------

class Gemma3Encoder:
    """Gemma 3 12B text encoder for LTX-2.

    Loads Gemma 3 model + text_embedding_projection + video_embeddings_connector.

    Usage:
        encoder = Gemma3Encoder(dtype=torch.bfloat16, device="cuda")
        encoder.load_gemma("/path/to/gemma_3_12B_it.safetensors")
        encoder.load_connector(checkpoint_sd)  # from LTX2 checkpoint
        hidden = encoder.encode("a cat playing")
        encoder.unload()
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.dtype = dtype
        self.device = torch.device(device)
        self.model: Gemma3Model | None = None
        self.text_projection: nn.Linear | None = None
        self.video_connector: EmbeddingsConnector | None = None
        self.tokenizer = None

    def load_gemma(self, path: str):
        """Load Gemma 3 12B weights from safetensors."""
        logger.info("Loading Gemma 3 12B from %s", path)
        from safetensors.torch import load_file
        import os

        if os.path.isdir(path):
            # Sharded checkpoint
            import glob
            shard_files = sorted(glob.glob(os.path.join(path, "model-*.safetensors")))
            sd = {}
            for sf in shard_files:
                sd.update(load_file(sf, device="cpu"))
        else:
            sd = load_file(path, device="cpu")

        # Strip 'model.' prefix if keys start with it
        config = Gemma3Config()

        with torch.device("meta"):
            self.model = Gemma3Model(config)

        # Map keys: file has 'model.layers.X...' and 'model.embed_tokens...'
        model_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                model_sd[k[6:]] = v  # strip 'model.'
            # Skip vision_model, lm_head etc.

        # Also handle norm key
        if "model.norm.weight" in sd:
            model_sd["norm.weight"] = sd["model.norm.weight"]

        missing, unexpected = self.model.load_state_dict(model_sd, strict=False, assign=True)
        logger.info("Gemma loaded: %d missing, %d unexpected", len(missing), len(unexpected))

        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        del sd, model_sd
        gc.collect()

    def load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from directory containing tokenizer.json or tokenizer.model."""
        import os
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded from %s (padding_side=left)", tokenizer_path)

    def load_connector(self, sd: dict[str, torch.Tensor]):
        """Load text_embedding_projection and video_embeddings_connector from LTX2 checkpoint.

        Args:
            sd: Full LTX2 checkpoint state dict (with or without prefixes)
        """
        # Text embedding projection
        proj_key = "text_embedding_projection.aggregate_embed.weight"
        if proj_key in sd:
            proj_w = sd[proj_key]
        elif f"model.diffusion_model.{proj_key}" in sd:
            # Shouldn't happen for this key but check anyway
            proj_w = sd[f"model.diffusion_model.{proj_key}"]
        else:
            raise KeyError(f"Cannot find {proj_key} in state dict")

        self.text_projection = nn.Linear(proj_w.shape[1], proj_w.shape[0], bias=False)
        self.text_projection.weight = nn.Parameter(proj_w.to(dtype=self.dtype))

        # Video embeddings connector
        prefix = "model.diffusion_model.video_embeddings_connector."
        conn_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                conn_sd[k[len(prefix):]] = v
            elif k.startswith("video_embeddings_connector."):
                conn_sd[k[len("video_embeddings_connector."):]] = v

        self.video_connector = EmbeddingsConnector(
            inner_dim=3840, num_heads=30, head_dim=128,
            num_layers=2, num_registers=128,
            max_pos=[1], theta=10000.0,
        )

        with torch.device("meta"):
            # Re-init on meta, then load
            pass

        missing, unexpected = self.video_connector.load_state_dict(conn_sd, strict=False, assign=True)
        if missing:
            real_missing = [k for k in missing if "rms_norm" not in k.lower()]
            if real_missing:
                logger.warning("Connector missing: %s", real_missing[:5])

        self.video_connector = self.video_connector.to(dtype=self.dtype)
        self.video_connector.eval()
        logger.info("Text projection + video connector loaded")

    def to_device(self):
        """Move encoder components to GPU."""
        if self.model is not None:
            self.model = self.model.to(self.device)
        if self.text_projection is not None:
            self.text_projection = self.text_projection.to(self.device)
        if self.video_connector is not None:
            self.video_connector = self.video_connector.to(self.device)

    def unload(self):
        """Move everything to CPU and free VRAM."""
        if self.model is not None:
            self.model = self.model.to("cpu")
        if self.text_projection is not None:
            self.text_projection = self.text_projection.to("cpu")
        if self.video_connector is not None:
            self.video_connector = self.video_connector.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, text: str, max_length: int = 1024) -> torch.Tensor:
        """Encode text to DiT-ready embeddings.

        Pipeline:
        1. Tokenize
        2. Gemma forward (all hidden states)
        3. Stack + project: 49 * 3840 -> 3840
        4. Normalize
        5. Video connector
        6. Output: (1, seq_len, 3840)

        The output needs to go through the DiT's caption_projection (3840 -> 4096).
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")

        # 1. Tokenize (left-padded to fixed max_length, matching reference)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 2. Forward through Gemma (all hidden states)
        all_hidden = self.model(input_ids, attention_mask)
        # all_hidden: list of 49 tensors, each (B, T, 3840)

        # 3. Stack all hidden states and project
        # Shape: (B, T, 49, 3840) -> movedim -> (B, 49, T, 3840) -> reshape (B, T, 49*3840)
        stacked = torch.stack(all_hidden, dim=1)  # (B, 49, T, 3840)
        stacked = stacked.movedim(1, -1)  # (B, T, 3840, 49)

        # 4. Normalize (padding-aware, matching reference)
        out = stacked.to(self.device)
        if attention_mask is not None:
            # mask shape: (B, T) -> (B, T, 1, 1)
            mask = attention_mask.bool().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)

            # Masked mean
            masked_out = out.masked_fill(~mask, 0.0)
            # Count of valid elements per batch: sum of mask over T * D dimensions
            seq_lens = attention_mask.sum(dim=1)  # (B,)
            d = out.shape[2]  # 3840
            denom = (seq_lens * d).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            mean = masked_out.sum(dim=(1, 2), keepdim=True) / (denom + 1e-6)

            # Masked min/max
            x_min = out.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
            x_max = out.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
            range_ = x_max - x_min

            out = 8.0 * (out - mean) / (range_ + 1e-6)
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (B, T, 188160)

            # Zero out padded positions
            mask_flat = attention_mask.bool().unsqueeze(-1)  # (B, T, 1)
            out = out.masked_fill(~mask_flat, 0.0)
        else:
            out = 8.0 * (out - out.mean(dim=(1, 2), keepdim=True)) / (
                out.amax(dim=(1, 2), keepdim=True) - out.amin(dim=(1, 2), keepdim=True) + 1e-6
            )
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (B, T, 188160)

        # 5. Project to 3840
        out = F.linear(out.float(), self.text_projection.weight.float()).to(self.dtype)  # (B, T, 3840)

        # 6. Video connector (pass attention mask for register replacement)
        # Build additive mask for connector: (B, 1, 1, T) with large negatives for padded
        connector_mask = None
        if attention_mask is not None:
            connector_mask = (attention_mask[:, None, None, :].to(self.dtype) - 1.0) * torch.finfo(self.dtype).max
        out, connector_mask = self.video_connector(out, connector_mask)

        return out
