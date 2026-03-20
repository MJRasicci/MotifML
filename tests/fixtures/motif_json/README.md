# IR Fixture Corpus

This directory contains the tracked raw Motif JSON fixtures used to exercise
MotifML's canonical IR surface.

The authoritative fixture catalog lives at `tests/fixtures/ir_fixture_catalog.json`.
Golden IR artifacts are tracked for a small subset and remain in `pending_review`
until the catalog marks them `approved`.

Regenerate these fixtures intentionally with:

```bash
uv run python tools/regenerate_ir_fixture_corpus.py
```

The generator owns the JSON fixture set under `tests/fixtures/motif_json/`,
`tests/fixtures/ir/golden/`, and `tests/fixtures/ir_fixture_catalog.json`. It
preserves any existing golden-review statuses while rewriting the generated
artifacts, so regeneration stays safe after review approval.

Fixtures:

## `single_track_monophonic_pickup`

A lead-guitar pickup phrase with a tied continuation and an explicit rest.

- Raw fixture: `tests/fixtures/motif_json/single_track_monophonic_pickup.json`
- Golden IR: `tests/fixtures/ir/golden/single_track_monophonic_pickup.ir.json`
- Proves:
  - `single-track monophonic`
  - `rests`
  - `ties`
  - `pickup bars`

## `ensemble_polyphony_controls`

A clarinet-plus-piano excerpt with transposition, multi-staff writing,
polyphony, and authored point/span controls.

- Raw fixture: `tests/fixtures/motif_json/ensemble_polyphony_controls.json`
- Golden IR: `tests/fixtures/ir/golden/ensemble_polyphony_controls.ir.json`
- Proves:
  - `multi-track polyphonic`
  - `tempo changes`
  - `dynamics`
  - `fermatas`
  - `hairpins`
  - `ottava`
  - `transposed instruments`
  - `multi-staff parts`

## `guitar_techniques_tuplets`

A single-staff guitar line with grace notes, triplets, linked-note techniques,
bends, and harmonics.

- Raw fixture: `tests/fixtures/motif_json/guitar_techniques_tuplets.json`
- Proves:
  - `tuplets`
  - `hammer-on/pull-off`
  - `slide links`
  - `bends`
  - `harmonics`
  - `grace notes`

## `voice_reentry`

A keyboard texture where the second written voice drops out for a bar and later
reappears.

- Raw fixture: `tests/fixtures/motif_json/voice_reentry.json`
- Proves:
  - `disappearing/reappearing voices`
