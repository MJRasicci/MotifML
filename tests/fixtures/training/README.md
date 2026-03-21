# Training Fixture Slice

This directory tracks the approved tiny training fixture slice and the generated
training-phase golden artifacts derived from it.

The fixture definition lives in `tests/fixtures/training/training_fixture_catalog.json`.
It is intentionally smaller than the full IR fixture corpus so training-preparation,
loader, and CPU smoke tests stay fast even if the broader raw fixture set grows.

Regenerate the tracked training-preparation artifacts intentionally with:

```bash
uv run python tools/regenerate_training_fixtures.py
```

Current training objectives covered by this slice:

- `split determinism`
- `vocabulary determinism`
- `tokenization validation`
- `loader behavior`
- `CPU smoke training`

Fixtures:

## `ensemble_polyphony_controls`

Multi-part polyphony with structure boundaries and authored control events.

- Raw fixture: `tests/fixtures/motif_json/ensemble_polyphony_controls.json`
- Proves:
  - `split determinism`
  - `tokenization validation`
  - `loader behavior`

## `guitar_techniques_tuplets`

Tuplets and guitar techniques that create a richer training-token surface.

- Raw fixture: `tests/fixtures/motif_json/guitar_techniques_tuplets.json`
- Proves:
  - `vocabulary determinism`
  - `tokenization validation`
  - `CPU smoke training`

## `single_track_monophonic_pickup`

A short pickup phrase with rests and ties for short-document window behavior.

- Raw fixture: `tests/fixtures/motif_json/single_track_monophonic_pickup.json`
- Proves:
  - `vocabulary determinism`
  - `loader behavior`
  - `CPU smoke training`

## `voice_reentry`

Voice-lane dropout and reentry that stress sequence ordering and replay.

- Raw fixture: `tests/fixtures/motif_json/voice_reentry.json`
- Proves:
  - `split determinism`
  - `loader behavior`
  - `CPU smoke training`

Generated artifact families in this directory:

- `split_manifest.json` and `split_stats.json` for deterministic split regression
- `vocabulary.json`, `vocabulary_version.json`, `vocab_stats.json`, and
  `model_input_stats.json` for frozen training-prep reporting
- `model_input/parameters.json`, `model_input/model_input_version.json`, and
  `model_input/storage_schema.json` for the frozen `05_model_input` contract metadata
- `representative_rows/` for human-reviewable tokenized document-row snapshots
- a normalized tiny-run smoke bundle for baseline training/evaluation regression

The tracked generated artifacts are owned by repository tools and should be regenerated
intentionally rather than hand-edited.
