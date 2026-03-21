Inspection and Regression Artifacts
===================================

This document describes the tracked artifact surfaces that MotifML uses to inspect,
debug, and regression-test IR and baseline-training changes.

Why These Artifacts Exist
-------------------------

MotifML's IR and baseline training stack both expose persisted contracts that are easier
to review when maintainers can inspect concrete artifact diffs. The repository therefore
keeps a small set of deterministic inspection artifacts in git so changes to persisted
IR shape, training-preparation surfaces, and baseline reporting behavior can be examined
directly.

The tracked surfaces are intended to support engineering analysis:

- fixture-backed coverage of representative symbolic constructs
- byte-stable golden IR artifacts for a curated subset of fixtures
- tabular and SVG inspection bundles for selected high-information examples
- tracked training-preparation fixtures and a normalized tiny smoke bundle
- notebook-based exploration for ad hoc inspection outside the production pipeline

Tracked Artifact Surfaces
-------------------------

The current repository tracks these artifact families:

``tests/fixtures/motif_json/``
   Curated raw Motif JSON fixtures that exercise the supported conversion surface.

``tests/fixtures/ir/golden/``
   Canonical persisted IR documents for a subset of fixtures. These are useful for
   spotting document-shape changes directly in JSON diffs.

``tests/fixtures/ir/inspection_bundles/``
   Deterministic inspection bundles generated from tracked fixtures. Each bundle includes
   the serialized IR document, schema-validation output, IR-validation output,
   structural summaries, CSV tables, and static SVG visualizations.

``tests/fixtures/ir_fixture_catalog.json``
   Fixture metadata, coverage mapping, and paths to generated artifacts.

``notebooks/ir_inspection.ipynb``
   Exploratory inspection surface for developers who want an interactive view without
   moving production logic out of ``src/motifml/``.

``tests/fixtures/training/``
   The approved tiny training fixture slice, tracked split/vocabulary/model-input
   metadata, representative tokenized rows, and the normalized baseline training and
   evaluation smoke bundle.

``notebooks/model_input_inspection.ipynb``
   Interactive inspection surface for split planning, vocabulary metadata, persisted
   tokenized rows, and reconstructed window views.

``notebooks/tokenization_validation.ipynb``
   Trace surface that walks one document from normalized IR through sequence projection
   and frozen-vocabulary encoding, then compares the result with the persisted row.

``notebooks/training_run_review.ipynb``
   Review surface for training curves, evaluation curves, decoded qualitative samples,
   and Markdown report consistency.

``notebooks/training_failure_analysis.ipynb``
   Investigation surface for unknown-token usage, vocabulary coverage, structural
   failures, and document pathologies surfaced by the persisted artifacts.

Generating Artifacts
--------------------

Use the repository tools to regenerate tracked artifacts:

.. code-block:: bash

   uv run python tools/regenerate_ir_fixture_corpus.py
   uv run python tools/generate_ir_inspection_bundles.py
   uv run python tools/regenerate_training_fixtures.py

The fixture-corpus generator rebuilds:

- raw Motif JSON fixtures
- the golden IR subset
- the fixture catalog

The inspection-bundle generator rebuilds the checked-in bundle directories under
``tests/fixtures/ir/inspection_bundles/``. You can also target specific fixtures with:

.. code-block:: bash

   uv run python tools/generate_ir_inspection_bundles.py --fixture-id <fixture_id>

``tools/regenerate_training_fixtures.py`` rebuilds the tracked training fixture slice
outputs under ``tests/fixtures/training/``:

- ``split_manifest.json`` and ``split_stats.json``
- ``vocabulary.json``, ``vocabulary_version.json``, and ``vocab_stats.json``
- ``model_input/`` metadata plus ``model_input_stats.json``
- ``representative_rows/`` tokenized-row snapshots
- ``smoke_bundle/`` training history, metrics, run metadata, decoded samples, and the
  qualitative Markdown report

Inspection Bundle Contents
--------------------------

Each generated bundle currently contains:

- ``README.md`` with a compact document summary
- ``bundle_manifest.json`` with bundle metadata
- ``source_identity.json`` linking the bundle back to its source fixture
- ``ir_document.ir.json`` with canonical serialized IR
- ``schema_validation.json`` with JSON Schema validation results
- ``ir_validation_report.json`` with rule-based structural validation output
- ``structural_summary.json`` with document-level counts and rollups
- ``voice_lane_onsets.csv`` with onset-level timing and voice-lane structure
- ``onset_notes.csv`` with note-level detail grouped by onset
- ``control_events.csv`` with point and span control data
- ``timeline_plot.svg`` showing the written timeline with per-bar counts
- ``voice_lane_ladder.svg`` showing voice-lane continuity and onset placement
- ``note_relations.svg`` showing tie and technique relations
- ``control_timeline.svg`` showing control events over written time

These artifacts are generated deterministically from the IR build and validation
pipelines. They are intended to make structural changes legible without relying on
temporary local outputs.

Training Fixture Contents
-------------------------

The tracked training fixture directory complements the IR bundles with a tiny
high-signal baseline-training surface.

It intentionally captures:

- deterministic split planning on a small approved corpus slice
- frozen vocabulary contents and version metadata
- frozen ``05_model_input`` metadata plus representative tokenized-document rows
- one normalized CPU smoke run covering both training and evaluation reporting

The four section 15 notebooks are expected to work against either these tracked
fixtures or real runtime outputs under ``data/``. For temporary integration runs, you
can point them at a generated artifact root with ``MOTIFML_TRAINING_ARTIFACT_ROOT``.

What to Inspect When IR Changes
-------------------------------

When a change affects IR mapping, serialization, validation, or projection-visible
structure, the most useful inspection surfaces are usually:

- the changed JSON under ``tests/fixtures/ir/golden/``
- the corresponding bundle README, validation report, tables, and SVGs
- the aggregate summary outputs produced by the ``ir_validation`` pipeline
- the fixture catalog coverage mapping if a new symbolic construct has been added

For changes that are intentionally non-structural, it is still helpful to say that no
persisted IR shape change was expected and confirm that the tracked artifacts remain
stable.

What to Inspect When Training Contracts Change
----------------------------------------------

When a change affects split planning, token naming, vocabulary reduction, model-input
metadata, lazy-loader behavior, training reporting, or baseline evaluation outputs, the
most useful inspection surfaces are usually:

- the diffs under ``tests/fixtures/training/``
- the representative rows under ``tests/fixtures/training/representative_rows/``
- the normalized smoke bundle under ``tests/fixtures/training/smoke_bundle/``
- the section 15 notebooks against either tracked fixtures or fresh runtime outputs

For intentional changes, say which persisted training surfaces were expected to move and
which were expected to remain stable.

Current Enforcement in Tests
----------------------------

The repository already codifies parts of this artifact strategy in tests:

- ``tests/ir/test_fixture_corpus.py`` checks fixture coverage, schema validity, and the
  generated catalog contract
- ``tests/pipelines/test_ir_build_fixture_determinism.py`` checks byte-stable IR and
  manifest output across repeated and reordered builds
- ``tests/ir/test_inspection_bundles.py`` requires the checked-in inspection bundles to
  reproduce exactly
- ``tests/pipelines/test_ir_pipeline_integration.py`` checks end-to-end Kedro pipeline
  behavior across the implemented stages

In other words, these artifacts are part of the project's regression surface, not just
developer convenience files.

The repository now also codifies the training-side artifact strategy in tests:

- ``tests/pipelines/test_training_preparation_fixture_regression.py`` checks the tracked
  split, vocabulary, and model-input fixture surfaces
- ``tests/pipelines/test_training_smoke_bundle_regression.py`` checks the normalized
  smoke bundle
- ``tests/pipelines/test_training_end_to_end_determinism.py`` checks repeated and
  reordered end-to-end runs for deterministic training-preparation and smoke outputs
- ``tests/analysis/`` executes the section 15 notebooks against both tracked fixtures
  and runtime-style outputs
