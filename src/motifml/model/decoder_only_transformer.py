"""Decoder-only Transformer baseline for MotifML sequence training."""

from __future__ import annotations

import math

import torch
from torch import nn

from motifml.model.config import DecoderOnlyTransformerConfig, PositionalEncodingType

_BATCH_RANK = 2


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings scoped to one fixed context length."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self._context_length = config.context_length
        self.embedding = nn.Embedding(config.context_length, config.embedding_dim)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Add learned position embeddings to one rank-3 token-embedding tensor."""
        batch_size, sequence_length, embedding_dim = token_embeddings.shape
        if sequence_length > self._context_length:
            raise ValueError(
                "sequence_length exceeds the configured context length: "
                f"{sequence_length} > {self._context_length}."
            )
        positions = torch.arange(sequence_length, device=token_embeddings.device)
        position_embeddings = self.embedding(positions).view(
            1,
            sequence_length,
            embedding_dim,
        )
        return token_embeddings + position_embeddings.expand(batch_size, -1, -1)


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional encodings for one fixed context length."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self._context_length = config.context_length
        encoding = _build_sinusoidal_encoding(
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
        )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal position encodings to one rank-3 token-embedding tensor."""
        sequence_length = token_embeddings.shape[1]
        if sequence_length > self._context_length:
            raise ValueError(
                "sequence_length exceeds the configured context length: "
                f"{sequence_length} > {self._context_length}."
            )
        return token_embeddings + self.encoding[:sequence_length].unsqueeze(0)


class DecoderTransformerBlock(nn.Module):
    """One pre-norm causal self-attention block with a feedforward sublayer."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        causal_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one causal Transformer block over a batch of hidden states."""
        normalized_states = self.attention_norm(hidden_states)
        attended_states, _ = self.attention(
            normalized_states,
            normalized_states,
            normalized_states,
            attn_mask=causal_attention_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attended_states
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)

        feedforward_states = self.feedforward_norm(hidden_states)
        hidden_states = hidden_states + self.feedforward(feedforward_states)
        return _apply_attention_mask(hidden_states, attention_mask)


class DecoderOnlyTransformer(nn.Module):
    """Baseline decoder-only Transformer for next-token prediction."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = build_positional_encoding_module(config)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderTransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.vocabulary_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project rank-2 token ids into next-token logits."""
        _validate_model_inputs(
            input_ids, attention_mask, context_length=self.config.context_length
        )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)

        causal_attention_mask = build_causal_attention_mask(
            sequence_length=input_ids.shape[1],
            device=input_ids.device,
        )
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                causal_attention_mask=causal_attention_mask,
                attention_mask=attention_mask,
            )

        hidden_states = self.final_norm(hidden_states)
        hidden_states = _apply_attention_mask(hidden_states, attention_mask)
        return self.output_projection(hidden_states)


def build_positional_encoding_module(
    config: DecoderOnlyTransformerConfig,
) -> nn.Module:
    """Build the configured positional-encoding module for one model instance."""
    if config.positional_encoding is PositionalEncodingType.LEARNED:
        return LearnedPositionalEncoding(config)
    if config.positional_encoding is PositionalEncodingType.SINUSOIDAL:
        return SinusoidalPositionalEncoding(config)
    raise ValueError(
        f"Unsupported positional_encoding value: {config.positional_encoding!r}."
    )


def build_causal_attention_mask(
    *,
    sequence_length: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the upper-triangular mask that blocks attention to future tokens."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def _apply_attention_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


def _build_sinusoidal_encoding(
    *,
    context_length: int,
    embedding_dim: int,
) -> torch.Tensor:
    positions = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / embedding_dim)
    )
    encoding = torch.zeros(context_length, embedding_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def _validate_model_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    context_length: int,
) -> None:
    if input_ids.ndim != _BATCH_RANK:
        raise ValueError("input_ids must be a rank-2 tensor.")
    if input_ids.shape[1] > context_length:
        raise ValueError(
            "input_ids sequence length exceeds the configured context length: "
            f"{input_ids.shape[1]} > {context_length}."
        )
    if attention_mask is None:
        return
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match the input_ids shape.")
    if attention_mask.ndim != _BATCH_RANK:
        raise ValueError("attention_mask must be a rank-2 tensor.")


__all__ = [
    "DecoderOnlyTransformer",
    "build_causal_attention_mask",
    "build_positional_encoding_module",
]
