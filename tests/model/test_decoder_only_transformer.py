"""Tests for the baseline decoder-only Transformer implementation."""

from __future__ import annotations

import torch
from torch.testing import assert_close

from motifml.model import DecoderOnlyTransformer, build_decoder_only_transformer_config


def test_decoder_only_transformer_forward_returns_vocab_sized_logits() -> None:
    model = DecoderOnlyTransformer(_build_config(vocabulary_size=31))
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    attention_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]],
        dtype=torch.bool,
    )

    logits = model(input_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 4, 31)


def test_decoder_only_transformer_is_causal_with_respect_to_future_tokens() -> None:
    torch.manual_seed(17)
    model = DecoderOnlyTransformer(_build_config(vocabulary_size=23))
    model.eval()

    prefix_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    prefix_b = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)

    logits_a = model(prefix_a)
    logits_b = model(prefix_b)

    assert_close(logits_a[:, :3], logits_b[:, :3])


def test_decoder_only_transformer_produces_deterministic_logits_under_fixed_seed() -> (
    None
):
    config = _build_config(vocabulary_size=19)
    inputs = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)

    torch.manual_seed(1234)
    first_model = DecoderOnlyTransformer(config)
    torch.manual_seed(1234)
    second_model = DecoderOnlyTransformer(config)
    first_model.eval()
    second_model.eval()

    first_logits = first_model(inputs)
    second_logits = second_model(inputs)

    assert_close(first_logits, second_logits)


def test_decoder_only_transformer_keeps_masked_positions_finite() -> None:
    model = DecoderOnlyTransformer(_build_config(vocabulary_size=29))
    model.eval()

    input_ids = torch.tensor([[0, 0, 6, 7]], dtype=torch.long)
    attention_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)

    logits = model(input_ids, attention_mask=attention_mask)

    assert torch.isfinite(logits).all()


def _build_config(*, vocabulary_size: int) -> object:
    return build_decoder_only_transformer_config(
        {
            "architecture": "decoder_only_transformer",
            "embedding_dim": 16,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "positional_encoding": "learned",
        },
        vocabulary_size=vocabulary_size,
        context_length=8,
        pad_token_id=0,
    )
