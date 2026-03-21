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
- deterministic score-level split planning for downstream experiments
- baseline normalization, feature-extraction, and tokenization stages for downstream ML
  experiments
- baseline decoder-only Transformer training with Kedro-managed checkpoints and
  reporting artifacts

Generation and model evaluation pipelines are not yet implemented in this repository.
The Kedro stage layout now uses ``06_models`` for baseline training checkpoints while
``07_model_output`` remains reserved for later evaluation and generation work.

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
     -> data/03_primary/ir/normalized_ir_version.json
     -> data/02_intermediate/training/split_manifest.json
     -> data/04_feature/ir/parameters.json + data/04_feature/ir/records/**/*.feature.json
     -> data/08_reporting/training/split_stats.json
     -> data/05_model_input/ir/vocabulary.json + data/05_model_input/ir/vocabulary_version.json
     -> data/08_reporting/training/vocab_stats.json
     -> data/05_model_input/ir/parameters.json + data/05_model_input/ir/storage_schema.json
     -> data/05_model_input/ir/records/<split>/<shard_id>/**/*.model_input.parquet
     -> data/05_model_input/ir/model_input_version.json
     -> data/08_reporting/training/model_input_stats.json
     -> data/06_models/training/baseline/checkpoint_manifest.json + best_checkpoint.json
     -> data/06_models/training/baseline/checkpoints/*.pt
     -> data/06_models/training/baseline/model_config.json + training_config.json + run_metadata.json
     -> data/08_reporting/training/training_history.json
     -> data/08_reporting/training/training_run_metadata.json

The raw corpus is sourced from files under ``data/00_corpus/`` together with the Motif
CLI binary at ``tools/motif-cli``. ``MotifJsonCorpusDataset`` fingerprints the source
files and CLI binary, stores the build state at
``data/02_intermediate/ingestion/raw_motif_json_build_state.json``, and only rebuilds
``data/01_raw/motif_json/`` when those inputs change.

Kedro Pipelines
---------------

``src/motifml/pipeline_registry.py`` registers named stage pipelines together with shard
and reducer variants:

- ``ingestion`` builds a file-level manifest and aggregate summary for the raw Motif JSON
  corpus
- ``ir_build`` validates the raw Motif JSON surface, constructs bar timing, emits typed
  IR entity families, assembles canonical IR documents, and writes an IR manifest
- ``ir_validation`` runs rule-based structural checks over persisted IR documents and
  produces corpus-level reporting
- ``normalization`` is currently a deterministic passthrough baseline from canonical IR
  to ``03_primary`` normalized IR, but it now also persists explicit
  ``normalized_ir_version`` metadata and validates that training-specific fields have
  not leaked into the normalized artifact surface
- ``dataset_splitting`` assigns deterministic score-level train / validation / test
  membership from normalized IR and persists both ``split_manifest`` and
  ``split_stats`` artifacts
- ``feature_extraction`` projects normalized IR into sequence, graph, or hierarchical
  feature views according to ``params:feature_extraction`` and the frozen
  ``params:sequence_schema`` contract, and persists explicit ``feature_version`` plus
  ``sequence_schema_version`` metadata with the partitioned ``04_feature`` output
- ``tokenization`` counts training-split tokens, reduces a frozen vocabulary, and
  persists Parquet-backed tokenized document rows plus ``model_input`` reporting and
  version metadata according to ``params:sequence_schema``, ``params:vocabulary``,
  ``params:model_input``, and ``params:data_split``
- ``training`` consumes lazy ``05_model_input`` runtime handles, trains the baseline
  decoder-only Transformer, and persists checkpoints plus run-reporting artifacts
- ``baseline_training`` composes the default preprocessing stages with ``training`` so
  maintainers have one explicit command path for end-to-end baseline runs from
  ``data/00_corpus`` through ``06_models`` and ``08_reporting``
- ``tokenization_shard`` persists shard-local Parquet-backed tokenized rows after the
  shared frozen vocabulary has been reduced
- ``model_input_reduce`` merges shard-local ``model_input_version`` fragments and
  aggregate ``model_input_stats`` reporting into the corpus-level reporting surface

The default pipeline intentionally stops before training so routine preprocessing runs do
not absorb expensive modeling work implicitly. Small staging nodes ensure that IR build
depends on the completed ingestion summary and that ``baseline_training`` does not begin
until the shared ``05_model_input`` artifacts have been persisted.

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
   repository state it preserves the canonical IR unchanged while enforcing the
   task-agnostic ``03_primary`` contract and emitting ``normalized_ir_version``.

``dataset_splitting``
   Produces deterministic score-level experiment splits independent of execution
   sharding. The stage persists a reviewable ``split_manifest`` and aggregate
   ``split_stats`` reporting surface under Kedro.

``feature_extraction``
   Selects one of three projection families:

   - sequence projection
   - graph projection
   - hierarchical projection

   For the baseline sequence path, ``feature_extraction.sequence_mode`` now makes the
   intended sequence surface explicit instead of inferring it only from legacy
   event-family flags. Persisted feature parameters also capture
   ``normalized_ir_version``, ``sequence_schema_version``, and the derived
   ``feature_version`` so downstream training preparation can freeze the exact emitted
   sequence contract.

``tokenization``
   Freezes the vocabulary-backed ``05_model_input`` contract. The pipeline counts
   training-split tokens, reduces one frozen vocabulary, tokenizes each projected
   sequence document into integer ids plus deterministic window metadata, and persists
   Parquet-backed tokenized rows together with ``model_input_version`` and
   ``model_input_stats`` artifacts. Explicit token naming, BOS/EOS placement,
   unknown-token handling, and encode/decode helpers live under ``motifml.training`` so
   tokenization, inspection, notebook, and evaluation code reuse one shared
   interpretation layer.
   The default pipeline persists rows under a synthetic ``global`` shard partition,
   while shard-local execution preserves concrete shard ids under the same physical
   layout so one dataset contract serves both execution modes.

``training``
   Streams split-scoped token windows lazily from persisted ``05_model_input`` rows,
   builds the baseline decoder-only Transformer from frozen Kedro parameters, and writes
   model checkpoints plus training history and run metadata. The canonical
   single-command run path is ``uv run kedro run --pipeline=baseline_training``.

Configuration Surfaces
----------------------

Variable behavior is driven through Kedro configuration rather than hardcoded constants.
The main project surfaces are:

- ``conf/base/catalog.yml`` for dataset locations and the raw-corpus auto-build contract
- ``conf/base/parameters.yml`` for IR build metadata, validation severities, feature
  projection settings, normalization contract settings, and the shared training-phase
  parameter families for dataset splitting, vocabulary construction, and model-input
  persistence
- ``conf/local/`` for machine-specific or sensitive overrides that should not be
  committed

Examples of currently configured parameters include:

- fixed IR build metadata such as ``ir_schema_version`` and ``corpus_build_version``
- per-rule validation severities for the structural validator
- projection type, explicit sequence mode, and sequence-schema settings for feature
  extraction
- vocabulary thresholds plus special-token policy settings for tokenization
- context length, stride, padding strategy, reporting thresholds, and storage settings
  for model-input persistence
- model architecture, seed, device selection, optimizer settings, gradient clipping,
  epoch count, and learning-rate scheduling for baseline training

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
- the training layer provides one explicit end-to-end baseline run path without
  expanding ``__default__`` into a heavy modeling command

See :doc:`/guides/contributing` for project-wide contribution guidance,
:doc:`/guides/ir_engineering` for IR-specific engineering notes,
:doc:`/guides/inspection_artifacts` for tracked inspection surfaces, and
:doc:`/reference/ir_contract` plus :doc:`/reference/normalized_ir_contract` for the
implemented IR and normalized-IR contracts.
