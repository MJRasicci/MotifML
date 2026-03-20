MotifML Architecture
====================

This document describes the implemented repository architecture: what the current
pipelines do, how data moves through Kedro, and which modules own the core symbolic-music
interfaces.

Implemented Scope
-----------------

The repository currently covers the symbolic data-engineering side of the project:

- deterministic ingestion of source corpora into canonical Motif JSON
- canonical IR construction from raw Motif JSON
- structural IR validation and corpus-level reporting
- deterministic inspection artifacts for fixture-backed regression analysis
- baseline normalization, feature-extraction, and tokenization stages for downstream ML
  experiments

Training, generation, checkpoint management, and model evaluation pipelines are not yet
implemented in this repository. The Kedro stage layout reserves ``06_models`` and
``07_model_output`` for that later work, but the current codebase stops at
``05_model_input`` plus reporting under ``08_reporting``.

End-to-End Data Flow
--------------------

The shared Kedro catalog in ``conf/base/catalog.yml`` defines this repository-level data
flow:

.. code-block:: text

   data/00_corpus/
     -> Motif CLI auto-build during dataset load
     -> data/01_raw/motif_json/
     -> data/02_intermediate/ingestion/raw_motif_json_manifest.json
     -> data/08_reporting/ingestion/raw_motif_json_summary.json
     -> data/02_intermediate/ir/documents/*.ir.json
     -> data/02_intermediate/ir/motif_ir_manifest.json
     -> data/08_reporting/ir/motif_ir_validation_report.json
     -> data/08_reporting/ir/motif_ir_summary.json
     -> data/03_primary/ir/documents/*.ir.json
     -> data/04_feature/ir/ir_features.json
     -> data/05_model_input/ir/model_input.json

The raw corpus is sourced from files under ``data/00_corpus/`` together with the Motif
CLI binary at ``tools/motif-cli``. ``MotifJsonCorpusDataset`` fingerprints the source
files and CLI binary, stores the build state at
``data/02_intermediate/ingestion/raw_motif_json_build_state.json``, and only rebuilds
``data/01_raw/motif_json/`` when those inputs change.

Kedro Pipelines
---------------

``src/motifml/pipeline_registry.py`` registers six named pipelines and one default
composition:

- ``ingestion`` builds a file-level manifest and aggregate summary for the raw Motif JSON
  corpus
- ``ir_build`` validates the raw Motif JSON surface, constructs bar timing, emits typed
  IR entity families, assembles canonical IR documents, and writes an IR manifest
- ``ir_validation`` runs rule-based structural checks over persisted IR documents and
  produces corpus-level reporting
- ``normalization`` is currently a deterministic passthrough baseline from canonical IR
  to ``03_primary`` normalized IR
- ``feature_extraction`` projects normalized IR into sequence, graph, or hierarchical
  feature views according to ``params:feature_extraction``
- ``tokenization`` converts projected features into a deterministic baseline
  ``model_input`` dataset according to ``params:tokenization``

The default pipeline composes those stages into a single DAG. A small staging node
ensures that IR build depends on the completed ingestion summary without mutating the raw
corpus dataset.

Current Pipeline Responsibilities
---------------------------------

``ingestion``
   Materializes stable file metadata such as path, hash, title, artist, album, track
   count, and playback-bar count for each raw Motif JSON score.

``ir_build``
   Owns the canonical conversion from raw Motif JSON into the repository's typed IR.
   This stage also aggregates conversion diagnostics into a manifest so unsupported,
   malformed, or intentionally excluded source features remain visible at corpus scale.

``ir_validation``
   Applies rule-based structural checks and emits both per-document validation reports
   and aggregate scale metrics for the corpus.

``normalization``
   Exists as a stable pipeline boundary for future normalization work. In the current
   repository state it preserves the canonical IR unchanged.

``feature_extraction``
   Selects one of three projection families:

   - sequence projection
   - graph projection
   - hierarchical projection

``tokenization``
   Packages projected features into a deterministic baseline model-input surface. The
   current implementation records projection metadata and simple structural counts in a
   fixed-length token sequence; it should be treated as a baseline contract rather than a
   finalized modeling vocabulary.

Configuration Surfaces
----------------------

Variable behavior is driven through Kedro configuration rather than hardcoded constants.
The main project surfaces are:

- ``conf/base/catalog.yml`` for dataset locations and the raw-corpus auto-build contract
- ``conf/base/parameters.yml`` for IR build metadata, validation severities, feature
  projection settings, and tokenization parameters
- ``conf/local/`` for machine-specific or sensitive overrides that should not be
  committed

Examples of currently configured parameters include:

- fixed IR build metadata such as ``ir_schema_version`` and ``corpus_build_version``
- per-rule validation severities for the structural validator
- projection type and event inclusion settings for feature extraction
- vocabulary strategy, max sequence length, padding strategy, and time resolution for
  tokenization

Code Organization
-----------------

The repository's primary implementation areas are:

- ``src/motifml/ir/`` for canonical IR models, identifiers, serialization,
  validation, summaries, projections, and deterministic inspection utilities
- ``src/motifml/datasets/`` for Kedro datasets that load raw Motif JSON corpora and
  persist canonical IR corpora
- ``src/motifml/pipelines/`` for Kedro node and pipeline definitions
- ``tests/`` for unit tests, integration tests, and fixture-backed regression coverage
- ``tests/fixtures/`` for tracked raw fixtures, golden IR artifacts, and checked-in
  inspection bundles
- ``tools/`` for deterministic fixture and inspection-bundle regeneration scripts
- ``notebooks/`` for exploratory inspection and visualization work that does not belong
  in production pipeline code

Determinism and Reproducibility
-------------------------------

Determinism is a first-class architectural requirement, not a convenience feature. The
current implementation enforces it in several layers:

- stable IR identifiers and canonical sort keys in ``src/motifml/ir/ids.py``
- typed emission result models in ``src/motifml/pipelines/ir_build/models.py`` that
  normalize ordering before assembly
- canonical JSON serialization in ``src/motifml/ir/serialization.py``
- IR corpus persistence that avoids rewriting byte-identical documents in
  ``src/motifml/datasets/motif_ir_corpus_dataset.py``
- fixture-backed tests that assert byte stability across repeated runs and reordered
  inputs

Research Workflow Support
-------------------------

The current codebase is designed to support symbolic-ML research in a disciplined way:

- the IR layer preserves authored written structure with exact rational timing
- the validation layer makes invariants and source-feature gaps explicit
- the reporting layer produces machine-readable corpus summaries
- the projection layer exposes multiple downstream views without changing the canonical
  IR
- the tokenization layer provides a reproducible baseline model-input contract for early
  experiments

See :doc:`/guides/contributing` for project-wide contribution guidance,
:doc:`/guides/ir_engineering` for IR-specific engineering notes,
:doc:`/guides/inspection_artifacts` for tracked inspection surfaces, and
:doc:`/reference/ir_contract` for the current IR contract.
