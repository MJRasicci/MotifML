"""Notebook support for sequence-model continuation roundtrip experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.datasets.training_checkpoint_dataset import TrainingCheckpointDataset
from motifml.evaluation.sampling import generate_greedy_continuation
from motifml.model import DecoderOnlyTransformer
from motifml.model.config import DecoderOnlyTransformerConfig
from motifml.pipelines.feature_extraction.nodes import extract_features
from motifml.pipelines.ir_build.nodes import (
    assemble_ir_document,
    build_written_time_map,
    emit_bars,
    emit_intrinsic_edges,
    emit_note_events,
    emit_onset_groups,
    emit_parts_and_staves,
    emit_point_control_events,
    emit_span_control_events,
    emit_voice_lanes,
    validate_canonical_score_surface,
)
from motifml.pipelines.normalization.nodes import (
    build_normalized_ir_version,
    normalize_ir_corpus,
)
from motifml.training.token_codec import encode_projected_events_to_tokens
from motifml.training.training_loop import resolve_torch_device

PITCH_TOKEN_RE = re.compile(r"^NOTE_PITCH:([A-G])([#B]*)(-?\d+)$")
NOTE_DURATION_TOKEN_RE = re.compile(r"^NOTE_DURATION:(\d+)$")
TIME_SHIFT_TOKEN_RE = re.compile(r"^TIME_SHIFT:(\d+)$")
_WHOLE_NOTES_PER_QUARTER = 4


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Filesystem layout for one notebook testbed run."""

    artifact_root: Path
    source_json_path: Path
    source_motif_path: Path
    output_json_path: Path
    output_gp_path: Path
    verify_json_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedNote:
    """One generated pitch-duration pair."""

    pitch_token: str
    duration_token: str

    @property
    def pitch_label(self) -> str:
        """Return the compact pitch label without the family prefix."""
        return self.pitch_token.split(":", maxsplit=1)[1]

    @property
    def duration_ticks(self) -> int:
        """Return the generated duration payload in quantized ticks."""
        match = NOTE_DURATION_TOKEN_RE.match(self.duration_token)
        if match is None:
            raise ValueError(f"Unsupported duration token: {self.duration_token!r}.")
        return int(match.group(1))


@dataclass(frozen=True, slots=True)
class ParsedOnset:
    """One parsed onset group with its bar-local offset."""

    offset_ticks: int
    notes: tuple[GeneratedNote, ...]

    @property
    def note_count(self) -> int:
        """Return the note count carried by this onset group."""
        return len(self.notes)

    @property
    def note_labels(self) -> tuple[str, ...]:
        """Return the compact pitch labels in token order."""
        return tuple(note.pitch_label for note in self.notes)

    @property
    def max_end_tick(self) -> int:
        """Return the farthest note end reached by this onset."""
        if not self.notes:
            return self.offset_ticks
        return max(self.offset_ticks + note.duration_ticks for note in self.notes)


@dataclass(frozen=True, slots=True)
class ParsedBar:
    """One generated bar parsed into onset groups."""

    leading_time_shift_ticks: int
    raw_tokens: tuple[str, ...]
    onsets: tuple[ParsedOnset, ...]
    parse_error: str | None = None

    @property
    def note_count(self) -> int:
        """Return the total generated note count inside the bar."""
        return sum(onset.note_count for onset in self.onsets)

    @property
    def onset_count(self) -> int:
        """Return the onset count inside the bar."""
        return len(self.onsets)

    @property
    def raw_note_labels(self) -> tuple[str, ...]:
        """Return one flattened list of generated pitch labels."""
        return tuple(note.pitch_label for onset in self.onsets for note in onset.notes)

    @property
    def onset_offsets(self) -> tuple[int, ...]:
        """Return the bar-local onset offsets in token order."""
        return tuple(onset.offset_ticks for onset in self.onsets)


@dataclass(frozen=True, slots=True)
class OnsetTemplate:
    """One source onset template used to reorder generated notes."""

    beat_index: int
    beat_template: dict[str, Any]
    offset_ticks: int
    json_pitch_labels: tuple[str, ...]
    permutation: tuple[int, ...]

    @property
    def note_count(self) -> int:
        """Return the note count required by this onset template."""
        return len(self.json_pitch_labels)


@dataclass(frozen=True, slots=True)
class TrackTemplate:
    """One source-track bar template."""

    track_name: str
    measure_template: dict[str, Any]
    bar_duration_ticks: int
    onset_templates: tuple[OnsetTemplate, ...]

    @property
    def onset_count(self) -> int:
        """Return the onset count required by this track."""
        return len(self.onset_templates)


@dataclass(frozen=True, slots=True)
class TrackBarSummary:
    """Readable summary for one generated track-local bar."""

    track_name: str
    generated_onsets: tuple[ParsedOnset, ...]
    onset_summaries: tuple[str, ...]
    note_labels: tuple[str, ...]
    reaches_bar_end: bool
    max_end_tick: int


@dataclass(frozen=True, slots=True)
class GeneratedBarSummary:
    """Human-readable interpretation of one generated bar."""

    bar_index: int
    leading_time_shift_ticks: int
    onset_count: int
    note_count: int
    parse_error: str | None
    is_structurally_complete: bool
    exported: bool
    raw_onset_offsets: tuple[int, ...]
    raw_note_labels: tuple[str, ...]
    track_summaries: tuple[TrackBarSummary, ...] = ()
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SequenceTestbedResult:
    """Collected artifacts and summaries for one notebook run."""

    repo_root: Path
    input_score_path: Path
    requested_complete_bars: int
    token_oversampling_factor: int
    fail_if_insufficient_complete_bars: bool
    paths: ArtifactPaths
    source_score: dict[str, Any]
    source_tokens: tuple[str, ...]
    sampling_metadata: dict[str, Any]
    generated_tokens: tuple[str, ...]
    template_tracks: tuple[TrackTemplate, ...]
    generated_bar_summaries: tuple[GeneratedBarSummary, ...]
    accepted_complete_bar_count: int
    export_performed: bool
    export_reason: str
    export_stdout: str
    verify_stdout: str
    verified_score: dict[str, Any] | None


def run_sequence_testbed(
    *,
    repo_root: Path,
    input_score_path: str | Path,
    requested_complete_bars: int,
    token_oversampling_factor: int = 8,
    fail_if_insufficient_complete_bars: bool = True,
) -> SequenceTestbedResult:
    """Run one parameterized continuation experiment and export when possible."""
    normalized_repo_root = repo_root.resolve()
    normalized_input_score_path = _resolve_input_score_path(
        normalized_repo_root,
        input_score_path,
    )
    if requested_complete_bars <= 0:
        raise ValueError("requested_complete_bars must be positive.")
    if token_oversampling_factor <= 0:
        raise ValueError("token_oversampling_factor must be positive.")

    paths = _build_artifact_paths(
        normalized_repo_root,
        normalized_input_score_path,
        requested_complete_bars=requested_complete_bars,
    )
    paths.artifact_root.mkdir(parents=True, exist_ok=True)
    motif_cli_path = normalized_repo_root / "tools" / "motif-cli"

    _run_motif_cli(motif_cli_path, normalized_input_score_path, paths.source_json_path)
    _run_motif_cli(motif_cli_path, normalized_input_score_path, paths.source_motif_path)

    source_score = _load_json(paths.source_json_path)
    vocabulary, source_tokens = _build_source_tokens(
        normalized_repo_root,
        paths.source_json_path,
    )
    vocabulary_time_resolution = int(
        vocabulary["construction_parameters"]["time_resolution"]
    )
    template_tracks = _extract_track_templates(
        source_score,
        source_tokens,
        time_resolution=vocabulary_time_resolution,
    )

    sampling_metadata, generated_tokens = _sample_continuation_tokens(
        normalized_repo_root,
        source_tokens,
        vocabulary,
        requested_complete_bars=requested_complete_bars,
        token_oversampling_factor=token_oversampling_factor,
    )

    parsed_generated_bars = _parse_bars(generated_tokens)
    generated_bar_summaries = _build_generated_bar_summaries(
        parsed_generated_bars,
        template_tracks,
        requested_complete_bars=requested_complete_bars,
    )
    accepted_bars = tuple(
        summary
        for summary in generated_bar_summaries
        if summary.is_structurally_complete
    )[:requested_complete_bars]
    accepted_complete_bar_count = len(accepted_bars)

    export_performed = False
    export_reason = "not_exported"
    export_stdout = ""
    verify_stdout = ""
    verified_score: dict[str, Any] | None = None

    if accepted_complete_bar_count >= requested_complete_bars:
        continued_score = _apply_generated_bars_to_score(
            source_score,
            template_tracks,
            accepted_bars,
            vocabulary_time_resolution=vocabulary_time_resolution,
        )
        _save_json(paths.output_json_path, continued_score)
        export_stdout = _run_motif_cli(
            motif_cli_path,
            paths.output_json_path,
            paths.output_gp_path,
            "--source-score",
            paths.source_motif_path,
        )
        verify_stdout = _run_motif_cli(
            motif_cli_path,
            paths.output_gp_path,
            paths.verify_json_path,
        )
        verified_score = _load_json(paths.verify_json_path)
        export_performed = True
        export_reason = "exported_requested_complete_bars"
    elif fail_if_insufficient_complete_bars:
        export_reason = "insufficient_complete_bars_for_strict_export"
    else:
        export_reason = "insufficient_complete_bars_export_skipped"

    result = SequenceTestbedResult(
        repo_root=normalized_repo_root,
        input_score_path=normalized_input_score_path,
        requested_complete_bars=requested_complete_bars,
        token_oversampling_factor=token_oversampling_factor,
        fail_if_insufficient_complete_bars=fail_if_insufficient_complete_bars,
        paths=paths,
        source_score=source_score,
        source_tokens=source_tokens,
        sampling_metadata=sampling_metadata,
        generated_tokens=generated_tokens,
        template_tracks=template_tracks,
        generated_bar_summaries=generated_bar_summaries,
        accepted_complete_bar_count=accepted_complete_bar_count,
        export_performed=export_performed,
        export_reason=export_reason,
        export_stdout=export_stdout,
        verify_stdout=verify_stdout,
        verified_score=verified_score,
    )
    _save_json(paths.summary_path, _result_summary_payload(result))
    return result


def result_summary_lines(result: SequenceTestbedResult) -> tuple[str, ...]:
    """Render one compact human-readable summary for notebook output."""
    lines = [
        f"Input score: {result.input_score_path}",
        f"Requested complete bars: {result.requested_complete_bars}",
        f"Accepted complete bars: {result.accepted_complete_bar_count}",
        f"Export performed: {result.export_performed}",
        f"Export reason: {result.export_reason}",
        (
            "Sampling: "
            f"{result.sampling_metadata['generated_token_count']} generated tokens "
            f"on {result.sampling_metadata['device']} "
            f"from checkpoint {result.sampling_metadata['best_checkpoint']}"
        ),
        (
            "Template onset counts by track: "
            + ", ".join(
                f"{template.track_name}={template.onset_count}"
                for template in result.template_tracks
            )
        ),
        (
            "Template bar durations by track: "
            + ", ".join(
                f"{template.track_name}={template.bar_duration_ticks} ticks"
                for template in result.template_tracks
            )
        ),
    ]
    return tuple(lines)


def formatted_bar_summaries(
    result: SequenceTestbedResult,
) -> tuple[tuple[str, ...], ...]:
    """Render each generated bar as plain notebook-friendly text lines."""
    rendered: list[tuple[str, ...]] = []
    for summary in result.generated_bar_summaries:
        header = (
            f"Generated bar {summary.bar_index}: "
            f"leading_time_shift_ticks={summary.leading_time_shift_ticks}, "
            f"onsets={summary.onset_count}, "
            f"notes={summary.note_count}, "
            f"complete={summary.is_structurally_complete}, "
            f"exported={summary.exported}"
        )
        lines = [header]
        if summary.parse_error is not None:
            lines.append(f"  Parse error: {summary.parse_error}")
        if summary.rejection_reason is not None:
            lines.append(f"  Rejection reason: {summary.rejection_reason}")
        lines.append(f"  Raw onset offsets: {list(summary.raw_onset_offsets)}")
        lines.append(f"  Raw note labels: {list(summary.raw_note_labels)}")
        for track_summary in summary.track_summaries:
            lines.append(
                f"  {track_summary.track_name}: {list(track_summary.note_labels)}"
                f" (max_end_tick={track_summary.max_end_tick}, "
                f"reaches_bar_end={track_summary.reaches_bar_end})"
            )
            for onset_summary in track_summary.onset_summaries:
                lines.append(f"    {onset_summary}")
        rendered.append(tuple(lines))
    return tuple(rendered)


def _resolve_input_score_path(
    repo_root: Path,
    input_score_path: str | Path,
) -> Path:
    candidate = Path(input_score_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _build_artifact_paths(
    repo_root: Path,
    input_score_path: Path,
    *,
    requested_complete_bars: int,
) -> ArtifactPaths:
    try:
        relative_input = input_score_path.relative_to(repo_root)
        relative_stem = relative_input.with_suffix("")
    except ValueError:
        relative_stem = Path(input_score_path.stem)

    artifact_id = "__".join(relative_stem.parts)
    artifact_root = repo_root / "temp" / "roundtrip" / "sequence_testbed" / artifact_id
    base_stem = input_score_path.stem
    request_suffix = f"{requested_complete_bars}bars"
    return ArtifactPaths(
        artifact_root=artifact_root,
        source_json_path=artifact_root / f"{base_stem}.mapped.json",
        source_motif_path=artifact_root / f"{base_stem}.source.motif",
        output_json_path=artifact_root / f"{base_stem}.{request_suffix}.continued.json",
        output_gp_path=artifact_root / f"{base_stem}.{request_suffix}.continued.gp",
        verify_json_path=(
            artifact_root / f"{base_stem}.{request_suffix}.continued.roundtrip.json"
        ),
        summary_path=artifact_root / f"{base_stem}.{request_suffix}.summary.json",
    )


def _run_motif_cli(
    motif_cli_path: Path,
    *args: str | Path,
) -> str:
    completed = subprocess.run(
        [str(motif_cli_path), *(str(arg) for arg in args)],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_source_tokens(
    repo_root: Path,
    score_json_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    payload = score_json_path.read_bytes()
    score = json.loads(payload)
    document = MotifJsonDocument(
        relative_path=score_json_path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        file_size_bytes=len(payload),
        score=score,
    )
    documents = [document]
    validation = validate_canonical_score_surface(documents)
    written_time_map = build_written_time_map(documents, validation)
    part_staff_emissions = emit_parts_and_staves(documents, validation)
    bar_emissions = emit_bars(documents, written_time_map)
    voice_lane_emissions = emit_voice_lanes(
        documents,
        part_staff_emissions,
        bar_emissions,
    )
    onset_group_emissions = emit_onset_groups(
        documents,
        written_time_map,
        voice_lane_emissions,
    )
    note_event_emissions = emit_note_events(
        documents,
        written_time_map,
        voice_lane_emissions,
        onset_group_emissions,
    )
    intrinsic_edge_emissions = emit_intrinsic_edges(
        documents,
        part_staff_emissions,
        bar_emissions,
        voice_lane_emissions,
        onset_group_emissions,
        note_event_emissions,
    )
    point_control_emissions = emit_point_control_events(
        documents,
        written_time_map,
        part_staff_emissions,
        voice_lane_emissions,
    )
    span_control_emissions = emit_span_control_events(
        documents,
        written_time_map,
        part_staff_emissions,
        voice_lane_emissions,
    )
    ir_corpus = assemble_ir_document(
        documents,
        part_staff_emissions,
        bar_emissions,
        voice_lane_emissions,
        onset_group_emissions,
        note_event_emissions,
        point_control_emissions,
        span_control_emissions,
        intrinsic_edge_emissions,
        {
            "ir_schema_version": "1.0.0",
            "corpus_build_version": "sequence_testbed",
            "build_timestamp": "2026-03-22T00:00:00-04:00",
        },
    )
    normalized_ir_corpus = normalize_ir_corpus(ir_corpus)
    normalized_ir_version = build_normalized_ir_version(
        normalized_ir_corpus,
        {
            "contract_name": "motifml.normalized_ir",
            "contract_version": "1.0.0",
            "serialized_document_format": "motifml.ir.document",
            "normalization_strategy": "passthrough_v1",
            "allow_optional_overlays": True,
            "allow_optional_views": True,
            "task_agnostic_guarantees": [
                "stable_source_relative_identity",
                "task_agnostic_domain_truth",
                "no_model_specific_flattening",
                "no_model_specific_windowing",
            ],
            "forbidden_model_fields": [
                "attention_mask",
                "input_ids",
                "model_input_version",
                "padding_strategy",
                "split",
                "split_version",
                "target_ids",
                "token_count",
                "token_ids",
                "training_run_id",
                "vocabulary_version",
                "window_start_offsets",
            ],
        },
    )
    feature_set = extract_features(
        normalized_ir_corpus,
        normalized_ir_version,
        {"projection_type": "sequence", "sequence_mode": "baseline_v1"},
        {
            "schema_name": "baseline_sequence",
            "schema_mode": "baseline_v1",
            "note_payload_fields": ["pitch", "duration"],
            "structure_markers": {
                "enabled": True,
                "marker_kinds": [
                    "part",
                    "staff",
                    "bar",
                    "voice_lane",
                    "onset_group",
                ],
            },
            "controls": {
                "include_point_controls": True,
                "point_control_kinds": [
                    "tempo_change",
                    "dynamic_change",
                    "fermata",
                ],
                "include_span_controls": True,
                "span_control_kinds": ["hairpin", "ottava"],
            },
        },
    )
    vocabulary = _load_json(
        repo_root / "data" / "05_model_input" / "ir" / "vocabulary.json"
    )
    tokens = encode_projected_events_to_tokens(
        feature_set.records[0].projection.events,
        time_resolution=int(vocabulary["construction_parameters"]["time_resolution"]),
        ordering_context=score_json_path.name,
        note_payload_fields=["pitch", "duration"],
        special_token_policy={
            "bos": "document",
            "eos": "document",
            "padding_interaction": "outside_boundaries",
            "unknown_token_mapping": "map_to_unk",
        },
    )
    return vocabulary, tokens


def _sample_continuation_tokens(
    repo_root: Path,
    source_tokens: tuple[str, ...],
    vocabulary: dict[str, Any],
    *,
    requested_complete_bars: int,
    token_oversampling_factor: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    training_artifacts = TrainingCheckpointDataset(
        str(repo_root / "data" / "06_models" / "training" / "baseline")
    ).load()
    model_config = DecoderOnlyTransformerConfig(**training_artifacts["model_config"])
    model = DecoderOnlyTransformer(model_config)
    best_checkpoint_name = str(training_artifacts["best_checkpoint"]["checkpoint_name"])
    best_checkpoint_state = next(
        checkpoint["state"]
        for checkpoint in training_artifacts["checkpoints"]
        if str(checkpoint["checkpoint_name"]) == best_checkpoint_name
    )
    model.load_state_dict(best_checkpoint_state["model_state_dict"])
    device = resolve_torch_device("auto")
    model.to(device)

    source_parsed_bars = _parse_bars(source_tokens)
    last_source_bar = source_parsed_bars[-1]
    estimated_tokens_per_bar = max(len(last_source_bar.raw_tokens), 1)
    max_new_tokens = max(
        estimated_tokens_per_bar * requested_complete_bars * token_oversampling_factor,
        model_config.context_length,
    )

    token_to_id = vocabulary["token_to_id"]
    id_to_token = {int(token_id): token for token, token_id in token_to_id.items()}
    prompt_token_ids = tuple(int(token_to_id[token]) for token in source_tokens[:-1])
    generated_token_ids = generate_greedy_continuation(
        model,
        prompt_token_ids=prompt_token_ids[-model_config.context_length :],
        max_new_tokens=max_new_tokens,
        device=device,
        context_length=model_config.context_length,
        eos_token_id=int(token_to_id["<eos>"]),
    )
    generated_tokens = tuple(
        id_to_token[int(token_id)] for token_id in generated_token_ids
    )
    metadata = {
        "device": str(device),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_context_token_count": min(
            len(prompt_token_ids),
            model_config.context_length,
        ),
        "generated_token_count": len(generated_tokens),
        "context_length": model_config.context_length,
        "best_checkpoint": best_checkpoint_name,
        "max_new_tokens": max_new_tokens,
        "estimated_tokens_per_bar": estimated_tokens_per_bar,
    }
    return metadata, generated_tokens


def _parse_bars(  # noqa: PLR0912, PLR0915
    tokens: tuple[str, ...],
) -> tuple[ParsedBar, ...]:
    bars: list[ParsedBar] = []
    current_tokens: list[str] | None = None
    current_onsets: list[ParsedOnset] = []
    current_onset_notes: list[GeneratedNote] | None = None
    current_onset_offset_ticks = 0
    pending_bar_time_shift_ticks = 0
    current_leading_time_shift_ticks = 0
    pending_pitch_token: str | None = None

    def finalize_current_bar(error: str | None = None) -> None:
        nonlocal current_tokens
        nonlocal current_onsets
        nonlocal current_onset_notes
        nonlocal current_onset_offset_ticks
        nonlocal current_leading_time_shift_ticks
        nonlocal pending_pitch_token
        if current_tokens is None:
            return
        onset_payloads = list(current_onsets)
        parse_error = error
        if pending_pitch_token is not None and parse_error is None:
            parse_error = "bar ended with NOTE_PITCH token missing NOTE_DURATION."
        if current_onset_notes is not None:
            onset_payloads.append(
                ParsedOnset(
                    offset_ticks=current_onset_offset_ticks,
                    notes=tuple(current_onset_notes),
                )
            )
        bars.append(
            ParsedBar(
                leading_time_shift_ticks=current_leading_time_shift_ticks,
                raw_tokens=tuple(current_tokens),
                onsets=tuple(onset_payloads),
                parse_error=parse_error,
            )
        )
        current_tokens = None
        current_onsets = []
        current_onset_notes = None
        current_onset_offset_ticks = 0
        current_leading_time_shift_ticks = 0
        pending_pitch_token = None

    for token in tokens:
        if token in {"<bos>", "<eos>"}:
            continue
        if token.startswith("TIME_SHIFT:"):
            shift_ticks = _ticks_from_time_shift_token(token)
            if current_tokens is None:
                pending_bar_time_shift_ticks += shift_ticks
                continue
            current_tokens.append(token)
            if pending_pitch_token is not None:
                finalize_current_bar("encountered TIME_SHIFT before NOTE_DURATION.")
                pending_bar_time_shift_ticks = 0
                continue
            if current_onset_notes is not None:
                current_onsets.append(
                    ParsedOnset(
                        offset_ticks=current_onset_offset_ticks,
                        notes=tuple(current_onset_notes),
                    )
                )
                current_onset_notes = None
            current_onset_offset_ticks += shift_ticks
            continue
        if token == "STRUCTURE:BAR":
            finalize_current_bar()
            current_leading_time_shift_ticks = pending_bar_time_shift_ticks
            pending_bar_time_shift_ticks = 0
            current_tokens = [token]
            current_onsets = []
            current_onset_notes = None
            current_onset_offset_ticks = 0
            pending_pitch_token = None
            continue
        if current_tokens is None:
            continue

        current_tokens.append(token)
        if token == "STRUCTURE:ONSET_GROUP":
            if pending_pitch_token is not None:
                finalize_current_bar(
                    "encountered STRUCTURE:ONSET_GROUP before NOTE_DURATION."
                )
                pending_bar_time_shift_ticks = 0
                continue
            if current_onset_notes is not None:
                current_onsets.append(
                    ParsedOnset(
                        offset_ticks=current_onset_offset_ticks,
                        notes=tuple(current_onset_notes),
                    )
                )
            current_onset_notes = []
            continue

        if token.startswith("NOTE_PITCH:"):
            if current_onset_notes is None:
                current_onset_notes = []
            if pending_pitch_token is not None:
                finalize_current_bar("encountered NOTE_PITCH before NOTE_DURATION.")
                pending_bar_time_shift_ticks = 0
                continue
            pending_pitch_token = token
            continue

        if token.startswith("NOTE_DURATION:"):
            if pending_pitch_token is None:
                finalize_current_bar("encountered NOTE_DURATION without NOTE_PITCH.")
                pending_bar_time_shift_ticks = 0
                continue
            if current_onset_notes is None:
                current_onset_notes = []
            current_onset_notes.append(
                GeneratedNote(
                    pitch_token=pending_pitch_token,
                    duration_token=token,
                )
            )
            pending_pitch_token = None

    finalize_current_bar()
    return tuple(bars)


def _extract_track_templates(
    source_score: dict[str, Any],
    source_tokens: tuple[str, ...],
    *,
    time_resolution: int,
) -> tuple[TrackTemplate, ...]:
    parsed_source_bars = _parse_bars(source_tokens)
    if not parsed_source_bars:
        raise ValueError("source score produced no parsed source bars.")
    last_source_bar = parsed_source_bars[-1]
    if last_source_bar.parse_error is not None:
        raise ValueError(
            "The source score's last bar could not be parsed into a stable token "
            f"template: {last_source_bar.parse_error}"
        )
    bar_duration_ticks = _ticks_from_score_time(
        source_score["timelineBars"][-1]["duration"],
        time_resolution=time_resolution,
    )

    source_onset_offset = 0
    track_templates: list[TrackTemplate] = []
    for track in source_score["tracks"]:
        measure_template = copy.deepcopy(track["staves"][0]["measures"][-1])
        voice_beats = measure_template["voices"][0]["beats"]
        note_beat_indices = [
            index for index, beat in enumerate(voice_beats) if beat.get("notes")
        ]
        onset_templates: list[OnsetTemplate] = []
        for beat_index in note_beat_indices:
            beat_template = copy.deepcopy(voice_beats[beat_index])
            source_onset = last_source_bar.onsets[source_onset_offset]
            source_onset_offset += 1
            template_pitch_labels = tuple(
                _pitch_label_from_note(note) for note in beat_template["notes"]
            )
            source_pitch_labels = source_onset.note_labels
            if len(template_pitch_labels) != len(source_pitch_labels):
                raise ValueError(
                    "The source score's last bar note count does not align with its "
                    "parsed token template. The current strict testbed expects one "
                    "parsable onset payload per note-bearing beat, so simultaneous "
                    "multi-track onset groups may still be unsupported."
                )
            template_offset_ticks = _ticks_from_score_time(
                beat_template["offset"],
                time_resolution=time_resolution,
            )
            if source_onset.offset_ticks != template_offset_ticks:
                raise ValueError(
                    "The source score's last bar onset offsets do not align with its "
                    "measure beat offsets."
                )
            onset_templates.append(
                OnsetTemplate(
                    beat_index=beat_index,
                    beat_template=beat_template,
                    offset_ticks=template_offset_ticks,
                    json_pitch_labels=template_pitch_labels,
                    permutation=_compute_permutation(
                        source_pitch_labels,
                        template_pitch_labels,
                    ),
                )
            )
        track_templates.append(
            TrackTemplate(
                track_name=str(track["name"]),
                measure_template=measure_template,
                bar_duration_ticks=bar_duration_ticks,
                onset_templates=tuple(onset_templates),
            )
        )

    if source_onset_offset != last_source_bar.onset_count:
        raise ValueError(
            "The source score's last bar onsets do not align with the parsed token "
            "template across tracks."
        )
    return tuple(track_templates)


def _build_generated_bar_summaries(
    parsed_generated_bars: tuple[ParsedBar, ...],
    template_tracks: tuple[TrackTemplate, ...],
    *,
    requested_complete_bars: int,
) -> tuple[GeneratedBarSummary, ...]:
    expected_onset_count = sum(track.onset_count for track in template_tracks)
    exported_complete_bar_count = 0
    summaries: list[GeneratedBarSummary] = []
    for bar_index, parsed_bar in enumerate(parsed_generated_bars, start=1):
        rejection_reason: str | None = None
        track_summaries: list[TrackBarSummary] = []
        is_complete = parsed_bar.parse_error is None
        if is_complete and parsed_bar.onset_count != expected_onset_count:
            is_complete = False
            rejection_reason = (
                "generated onset count does not match the source last-bar template."
            )

        onset_offset = 0
        if is_complete:
            for track in template_tracks:
                generated_track_onsets = parsed_bar.onsets[
                    onset_offset : onset_offset + track.onset_count
                ]
                onset_offset += track.onset_count
                if len(generated_track_onsets) != track.onset_count:
                    is_complete = False
                    rejection_reason = (
                        "generated track-local onset count is incomplete."
                    )
                    break
                track_note_labels: list[str] = []
                onset_summaries: list[str] = []
                track_max_end_tick = 0
                for onset_template, generated_onset in zip(
                    track.onset_templates,
                    generated_track_onsets,
                    strict=True,
                ):
                    if generated_onset.offset_ticks != onset_template.offset_ticks:
                        is_complete = False
                        rejection_reason = (
                            "generated onset offsets do not match the source "
                            "last-bar template."
                        )
                        break
                    if generated_onset.note_count != onset_template.note_count:
                        is_complete = False
                        rejection_reason = (
                            "generated onset note count does not match the source "
                            "last-bar template."
                        )
                        break
                    reordered_onset_notes = tuple(
                        generated_onset.notes[index]
                        for index in onset_template.permutation
                    )
                    reordered_pitch_labels = [
                        generated_note.pitch_label
                        for generated_note in reordered_onset_notes
                    ]
                    track_note_labels.extend(reordered_pitch_labels)
                    track_max_end_tick = max(
                        track_max_end_tick,
                        *(
                            generated_onset.offset_ticks + note.duration_ticks
                            for note in reordered_onset_notes
                        ),
                    )
                    onset_summaries.append(
                        f"{_format_score_time_dict(onset_template.beat_template['offset'])}"
                        f" ({generated_onset.offset_ticks} ticks): "
                        f"{reordered_pitch_labels}"
                    )
                if not is_complete:
                    break
                reaches_bar_end = track_max_end_tick == track.bar_duration_ticks
                if not reaches_bar_end:
                    is_complete = False
                    rejection_reason = (
                        "generated durations do not fill the source bar duration "
                        "exactly."
                    )
                    break
                track_summaries.append(
                    TrackBarSummary(
                        track_name=track.track_name,
                        generated_onsets=tuple(
                            ParsedOnset(
                                offset_ticks=generated_onset.offset_ticks,
                                notes=tuple(
                                    generated_onset.notes[index]
                                    for index in onset_template.permutation
                                ),
                            )
                            for onset_template, generated_onset in zip(
                                track.onset_templates,
                                generated_track_onsets,
                                strict=True,
                            )
                        ),
                        onset_summaries=tuple(onset_summaries),
                        note_labels=tuple(track_note_labels),
                        reaches_bar_end=reaches_bar_end,
                        max_end_tick=track_max_end_tick,
                    )
                )

        exported = False
        if is_complete and exported_complete_bar_count < requested_complete_bars:
            exported = True
            exported_complete_bar_count += 1
        summaries.append(
            GeneratedBarSummary(
                bar_index=bar_index,
                leading_time_shift_ticks=parsed_bar.leading_time_shift_ticks,
                onset_count=parsed_bar.onset_count,
                note_count=parsed_bar.note_count,
                parse_error=parsed_bar.parse_error,
                is_structurally_complete=is_complete,
                exported=exported,
                raw_onset_offsets=parsed_bar.onset_offsets,
                raw_note_labels=parsed_bar.raw_note_labels,
                track_summaries=tuple(track_summaries),
                rejection_reason=rejection_reason,
            )
        )
    return tuple(summaries)


def _apply_generated_bars_to_score(
    source_score: dict[str, Any],
    template_tracks: tuple[TrackTemplate, ...],
    accepted_bars: tuple[GeneratedBarSummary, ...],
    *,
    vocabulary_time_resolution: int,
) -> dict[str, Any]:
    continued_score = copy.deepcopy(source_score)
    next_note_id, next_beat_id = _next_note_and_beat_ids(continued_score)

    for summary in accepted_bars:
        previous_timeline_bar = continued_score["timelineBars"][-1]
        new_bar_index = len(continued_score["timelineBars"])
        new_timeline_bar = copy.deepcopy(previous_timeline_bar)
        new_timeline_bar["index"] = new_bar_index
        new_timeline_bar["start"] = _add_rationals(
            previous_timeline_bar["start"],
            previous_timeline_bar["duration"],
        )
        new_timeline_bar["sectionLetter"] = ""
        new_timeline_bar["sectionText"] = ""
        new_timeline_bar["jump"] = ""
        new_timeline_bar["target"] = ""
        new_timeline_bar["repeatStart"] = False
        new_timeline_bar["repeatEnd"] = False
        new_timeline_bar["repeatCount"] = -1
        new_timeline_bar["alternateEndings"] = ""
        continued_score["timelineBars"].append(new_timeline_bar)
        continued_score["playbackMasterBarSequence"].append(new_bar_index)

        for track, track_template, track_summary in zip(
            continued_score["tracks"],
            template_tracks,
            summary.track_summaries,
            strict=True,
        ):
            new_measure = copy.deepcopy(track_template.measure_template)
            new_measure["index"] = new_bar_index
            new_voice_beats = new_measure["voices"][0]["beats"]
            generated_onset_offset = 0
            for beat in new_voice_beats:
                beat["id"] = next_beat_id
                next_beat_id += 1
                if not beat.get("notes"):
                    continue
                onset_template = track_template.onset_templates[generated_onset_offset]
                generated_onset = track_summary.generated_onsets[generated_onset_offset]
                new_notes: list[dict[str, Any]] = []
                for template_note, generated_note in zip(
                    onset_template.beat_template["notes"],
                    generated_onset.notes,
                    strict=True,
                ):
                    new_note = copy.deepcopy(template_note)
                    new_note["id"] = next_note_id
                    next_note_id += 1
                    new_note["pitch"] = _pitch_from_label(generated_note.pitch_label)
                    generated_duration = _score_time_from_ticks(
                        generated_note.duration_ticks,
                        time_resolution=vocabulary_time_resolution,
                    )
                    new_note["duration"] = generated_duration
                    new_note["soundingDuration"] = copy.deepcopy(generated_duration)
                    new_note["articulation"] = _neutral_articulation()
                    new_notes.append(new_note)
                beat["notes"] = new_notes
                generated_onset_offset += 1
            new_measure["beats"] = copy.deepcopy(new_measure["voices"][0]["beats"])
            track["staves"][0]["measures"].append(new_measure)

    return continued_score


def _result_summary_payload(result: SequenceTestbedResult) -> dict[str, Any]:
    return {
        "input_score_path": str(result.input_score_path),
        "requested_complete_bars": result.requested_complete_bars,
        "token_oversampling_factor": result.token_oversampling_factor,
        "fail_if_insufficient_complete_bars": (
            result.fail_if_insufficient_complete_bars
        ),
        "paths": {
            "artifact_root": str(result.paths.artifact_root),
            "source_json_path": str(result.paths.source_json_path),
            "source_motif_path": str(result.paths.source_motif_path),
            "output_json_path": str(result.paths.output_json_path),
            "output_gp_path": str(result.paths.output_gp_path),
            "verify_json_path": str(result.paths.verify_json_path),
            "summary_path": str(result.paths.summary_path),
        },
        "sampling_metadata": dict(result.sampling_metadata),
        "template_tracks": [
            {
                "track_name": template.track_name,
                "bar_duration_ticks": template.bar_duration_ticks,
                "onset_count": template.onset_count,
                "onsets": [
                    {
                        "beat_index": onset_template.beat_index,
                        "offset": onset_template.beat_template["offset"],
                        "offset_ticks": onset_template.offset_ticks,
                        "note_count": onset_template.note_count,
                        "json_pitch_labels": list(onset_template.json_pitch_labels),
                        "permutation": list(onset_template.permutation),
                    }
                    for onset_template in template.onset_templates
                ],
            }
            for template in result.template_tracks
        ],
        "generated_bar_summaries": [
            {
                "bar_index": summary.bar_index,
                "leading_time_shift_ticks": summary.leading_time_shift_ticks,
                "onset_count": summary.onset_count,
                "note_count": summary.note_count,
                "parse_error": summary.parse_error,
                "is_structurally_complete": summary.is_structurally_complete,
                "exported": summary.exported,
                "raw_onset_offsets": list(summary.raw_onset_offsets),
                "raw_note_labels": list(summary.raw_note_labels),
                "rejection_reason": summary.rejection_reason,
                "track_summaries": [
                    {
                        "track_name": track_summary.track_name,
                        "generated_onsets": [
                            {
                                "offset_ticks": onset.offset_ticks,
                                "notes": [
                                    {
                                        "pitch_label": note.pitch_label,
                                        "duration_ticks": note.duration_ticks,
                                    }
                                    for note in onset.notes
                                ],
                            }
                            for onset in track_summary.generated_onsets
                        ],
                        "onset_summaries": list(track_summary.onset_summaries),
                        "note_labels": list(track_summary.note_labels),
                        "reaches_bar_end": track_summary.reaches_bar_end,
                        "max_end_tick": track_summary.max_end_tick,
                    }
                    for track_summary in summary.track_summaries
                ],
            }
            for summary in result.generated_bar_summaries
        ],
        "accepted_complete_bar_count": result.accepted_complete_bar_count,
        "export_performed": result.export_performed,
        "export_reason": result.export_reason,
        "export_stdout": result.export_stdout,
        "verify_stdout": result.verify_stdout,
        "verified_bar_count": (
            None
            if result.verified_score is None
            else len(result.verified_score["timelineBars"])
        ),
    }


def _compute_permutation(
    source_pitch_labels: tuple[str, ...],
    template_pitch_labels: tuple[str, ...],
) -> tuple[int, ...]:
    positions_by_pitch: dict[str, deque[int]] = {}
    for index, pitch_label in enumerate(source_pitch_labels):
        positions_by_pitch.setdefault(pitch_label, deque()).append(index)
    permutation: list[int] = []
    for pitch_label in template_pitch_labels:
        try:
            permutation.append(positions_by_pitch[pitch_label].popleft())
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "Unable to derive one source-template permutation for onset notes."
            ) from exc
    return tuple(permutation)


def _neutral_articulation() -> dict[str, Any]:
    return {
        "leftFingering": "",
        "rightFingering": "",
        "ornament": "",
        "letRing": False,
        "vibrato": "",
        "tieOrigin": False,
        "tieDestination": False,
        "trill": None,
        "trillSpeed": 0,
        "accent": None,
        "antiAccent": False,
        "palmMuted": False,
        "muted": False,
        "tapped": False,
        "leftHandTapped": False,
        "hopoOrigin": False,
        "hopoDestination": False,
        "hopoType": 0,
        "slides": [],
        "harmonic": None,
        "bend": None,
        "relations": [],
    }


def _pitch_label_from_note(note: dict[str, Any]) -> str:
    pitch = note["pitch"]
    accidental = pitch.get("accidental") or ""
    accidental = accidental.replace("b", "B")
    return f"{pitch['step']}{accidental}{pitch['octave']}"


def _pitch_from_label(pitch_label: str) -> dict[str, Any]:
    match = PITCH_TOKEN_RE.match(f"NOTE_PITCH:{pitch_label}")
    if match is None:
        raise ValueError(f"Unsupported pitch label: {pitch_label!r}.")
    step, accidental, octave_text = match.groups()
    return {
        "step": step,
        "accidental": accidental.replace("B", "b").replace("N", ""),
        "octave": int(octave_text),
    }


def _ticks_from_time_shift_token(token: str) -> int:
    match = TIME_SHIFT_TOKEN_RE.match(token)
    if match is None:
        raise ValueError(f"Unsupported time-shift token: {token!r}.")
    return int(match.group(1))


def _format_score_time_dict(score_time: dict[str, int]) -> str:
    return f"{score_time['numerator']}/{score_time['denominator']}"


def _add_rationals(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    numerator = (
        left["numerator"] * right["denominator"]
        + right["numerator"] * left["denominator"]
    )
    denominator = left["denominator"] * right["denominator"]
    factor = gcd(numerator, denominator)
    return {
        "numerator": numerator // factor,
        "denominator": denominator // factor,
    }


def _next_note_and_beat_ids(score: dict[str, Any]) -> tuple[int, int]:
    max_note_id = -1
    max_beat_id = -1
    for track in score["tracks"]:
        for staff in track["staves"]:
            for measure in staff["measures"]:
                for voice in measure["voices"]:
                    for beat in voice["beats"]:
                        max_beat_id = max(max_beat_id, int(beat["id"]))
                        for note in beat["notes"]:
                            max_note_id = max(max_note_id, int(note["id"]))
    return max_note_id + 1, max_beat_id + 1


def _score_time_from_ticks(
    ticks: int,
    *,
    time_resolution: int,
) -> dict[str, int]:
    if ticks <= 0:
        raise ValueError("ticks must be positive.")
    denominator = time_resolution * _WHOLE_NOTES_PER_QUARTER
    factor = gcd(ticks, denominator)
    return {
        "numerator": ticks // factor,
        "denominator": denominator // factor,
    }


def _ticks_from_score_time(
    score_time: dict[str, int],
    *,
    time_resolution: int,
) -> int:
    numerator = int(score_time["numerator"])
    denominator = int(score_time["denominator"])
    if denominator <= 0:
        raise ValueError("score_time denominator must be positive.")
    scaled_numerator = numerator * time_resolution * _WHOLE_NOTES_PER_QUARTER
    if scaled_numerator % denominator != 0:
        raise ValueError(
            "score_time does not align with the configured time resolution."
        )
    return scaled_numerator // denominator


__all__ = [
    "ArtifactPaths",
    "GeneratedBarSummary",
    "SequenceTestbedResult",
    "formatted_bar_summaries",
    "result_summary_lines",
    "run_sequence_testbed",
]
