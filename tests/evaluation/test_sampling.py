"""Tests for deterministic qualitative sampling helpers."""

from __future__ import annotations

import torch
from torch import nn

from motifml.evaluation.sampling import (
    build_prompt_continuation_samples,
    coerce_loaded_tokenized_documents,
    generate_greedy_continuation,
    summarize_decoded_tokens,
)
from motifml.training.contracts import DatasetSplit
from motifml.training.model_input import TokenizedDocumentRow

VOCABULARY = {
    "token_to_id": {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "STRUCTURE:BAR": 4,
        "NOTE_PITCH:C4": 5,
        "NOTE_DURATION:96": 6,
        "NOTE_PITCH:D4": 7,
        "NOTE_PITCH:E4": 8,
    }
}


def test_generate_greedy_continuation_stops_after_eos() -> None:
    model = _TransitionLogitModel({6: 7, 7: 2, 2: 2})

    continuation = generate_greedy_continuation(
        model,
        prompt_token_ids=(1, 5, 6),
        max_new_tokens=4,
        device=torch.device("cpu"),
        context_length=6,
        eos_token_id=2,
    )

    assert continuation == (7, 2)


def test_build_prompt_continuation_samples_decodes_review_surfaces_deterministically() -> (
    None
):
    model = _TransitionLogitModel({6: 7, 7: 8, 8: 2, 2: 2})
    documents = coerce_loaded_tokenized_documents(
        (
            _build_row(
                relative_path="fixtures/a.json",
                document_id="doc-a",
                token_ids=(1, 4, 5, 6, 7, 2),
            ),
            _build_row(
                relative_path="fixtures/b.json",
                document_id="doc-b",
                token_ids=(1, 4, 5, 6, 8, 2),
            ),
        )
    )

    samples = build_prompt_continuation_samples(
        model,
        documents=documents,
        vocabulary=VOCABULARY,
        samples_per_split=1,
        prompt_token_count=4,
        continuation_token_count=3,
        summary_token_limit=2,
        device=torch.device("cpu"),
        context_length=6,
        eos_token_id=2,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.split is DatasetSplit.VALIDATION
    assert sample.relative_path == "fixtures/a.json"
    assert sample.prompt_tokens == (
        "<bos>",
        "STRUCTURE:BAR",
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
    )
    assert sample.reference_continuation_tokens == ("NOTE_PITCH:D4", "<eos>")
    assert sample.generated_continuation_tokens == (
        "NOTE_PITCH:D4",
        "NOTE_PITCH:E4",
        "<eos>",
    )
    assert sample.prompt_summary == "<bos> STRUCTURE:BAR ... (+2 more)"
    assert sample.generated_summary == "NOTE_PITCH:D4 NOTE_PITCH:E4 ... (+1 more)"


def test_summarize_decoded_tokens_renders_empty_and_truncated_sequences() -> None:
    assert summarize_decoded_tokens((), max_tokens=3) == "<empty>"
    assert (
        summarize_decoded_tokens(("A", "B", "C", "D"), max_tokens=2)
        == "A B ... (+2 more)"
    )


class _TransitionLogitModel(nn.Module):
    def __init__(self, transitions: dict[int, int], vocabulary_size: int = 9) -> None:
        super().__init__()
        self._transitions = dict(transitions)
        self._vocabulary_size = vocabulary_size

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self._vocabulary_size),
            fill_value=-1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for batch_index in range(batch_size):
            for position_index in range(sequence_length):
                current_token = int(input_ids[batch_index, position_index].item())
                next_token = self._transitions.get(current_token, 2)
                logits[batch_index, position_index, next_token] = 1000.0
        return logits


def _build_row(
    *,
    relative_path: str,
    document_id: str,
    token_ids: tuple[int, ...],
) -> TokenizedDocumentRow:
    return TokenizedDocumentRow(
        relative_path=relative_path,
        document_id=document_id,
        split=DatasetSplit.VALIDATION,
        split_version="split-v1",
        projection_type="sequence",
        sequence_mode="baseline_v1",
        normalized_ir_version="normalized-v1",
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_version="model-input-v1",
        storage_schema_version="parquet-v1",
        token_count=len(token_ids),
        token_ids=token_ids,
        window_start_offsets=(0,),
        context_length=6,
        stride=3,
        padding_strategy="right",
        special_token_policy={
            "bos": "document",
            "eos": "document",
            "padding_interaction": "outside_boundaries",
            "unknown_token_mapping": "map_to_unk",
        },
    )
