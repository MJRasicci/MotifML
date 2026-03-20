MotifML Architecture
====================

This document describes the architecture that exists in the repository today.

System Overview
---------------

MotifML currently implements the symbolic-data side of the project:

- ingestion of source corpora into canonical Motif JSON
- deterministic IR build and validation
- tracked review artifacts for fixture approval
- baseline normalization, feature extraction, and tokenization stages for downstream ML
  work

The repository does not yet contain training, generation, or evaluation pipelines. Those
remain future work and are intentionally not described here as implemented behavior.

Current Data Flow
-----------------

.. code-block:: text

   data/00_corpus
     -> motif-cli autobuild during ingestion
     -> data/01_raw/motif_json
     -> data/02_intermediate/ingestion/raw_motif_json_manifest.json
     -> data/08_reporting/ingestion/raw_motif_json_summary.json
     -> data/02_intermediate/ir/documents/*.ir.json
     -> data/02_intermediate/ir/motif_ir_manifest.json
     -> data/08_reporting/ir/motif_ir_validation_report.json
     -> data/08_reporting/ir/motif_ir_summary.json
     -> data/03_primary/ir/documents
     -> data/04_feature/ir/ir_features.json
     -> data/05_model_input/ir/model_input.json

The raw corpus is built from files under ``data/00_corpus`` plus the Motif CLI binary in
``tools/motif-cli``. Kedro stores the raw-build fingerprint in
``data/02_intermediate/ingestion/raw_motif_json_build_state.json`` and only rebuilds when
the corpus contents or CLI binary change.

Kedro Pipelines
---------------

``src/motifml/pipeline_registry.py`` currently registers these pipelines:

- ``ingestion``: builds and summarizes the raw Motif JSON corpus
- ``ir_build``: validates the Motif JSON canonical surface and emits IR documents plus a
  manifest
- ``ir_validation``: runs structural validation and produces corpus-level reporting
- ``normalization``: deterministic passthrough baseline from IR to normalized IR
- ``feature_extraction``: projection-driven baseline for sequence, graph, or
  hierarchical views
- ``tokenization``: deterministic baseline packaging from extracted features to model
  input

The default pipeline runs those stages in order. A small staging node gates IR build on a
completed ingestion summary without mutating the raw corpus.

Repository Layout
-----------------

The main implementation areas are:

- ``src/motifml/ir/``: canonical IR models, ids, serialization, validation, projections,
  summaries, and review-bundle utilities
- ``src/motifml/datasets/``: Kedro datasets for raw Motif JSON and persisted IR corpora
- ``src/motifml/pipelines/``: Kedro node and pipeline definitions
- ``tests/fixtures/``: tracked raw fixtures, golden IR artifacts, and checked-in review
  bundles
- ``tools/``: fixture and review-bundle regeneration entry points

Canonical IR Layer
------------------

The IR itself is a written-score representation with:

- structural entities: ``Part``, ``Staff``, ``Bar``, ``VoiceLane``
- event entities: ``OnsetGroup``, ``NoteEvent``, ``PointControlEvent``,
  ``SpanControlEvent``
- sparse intrinsic edges: ``contains``, ``next_in_voice``, ``tie_to``,
  ``technique_to``
- optional overlays and derived-view containers

Documents are serialized one-per-score as canonical JSON. The build manifest tracks
source-relative identity, hashes, node and edge counts, and grouped conversion
diagnostics outside the IR body.

Determinism and Review
----------------------

Determinism is a core architectural constraint:

- IR identifiers are built from stable source position using ``src/motifml/ir/ids.py``
- emission result models and serialization apply canonical ordering before persistence
- the IR corpus dataset avoids rewriting byte-identical documents
- fixture-backed tests assert byte stability across repeated builds and reordered inputs

Tracked review artifacts are part of the architecture, not an afterthought. The
repository tracks:

- curated raw fixtures and a golden IR subset in ``tests/fixtures/``
- review-bundle artifacts in ``tests/fixtures/ir/review_bundles/``
- approval status in ``tests/fixtures/ir_fixture_catalog.json``

See :doc:`ir_design` for the IR contract and :doc:`ir_contributor_guide` for the
expected contributor workflow.
