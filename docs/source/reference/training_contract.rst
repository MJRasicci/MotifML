Baseline Training Contract
==========================

This page summarizes the implemented baseline training contract for MotifML. It freezes
the main configuration families, version-key derivation rules, token and split
semantics, and the persisted artifact layout that maintainers should expect from the
baseline stack.

It is intentionally retained as the runnable baseline reference. The authoritative design
contract for the in-progress recovery path lives in :doc:`v1_continuation_task_contract`.
That page defines the future single-track, single-voice, next-bar continuation task;
this page documents the currently implemented document-row baseline.

Approved Baseline Boundary
--------------------------

The approved v1 baseline path is:

- normalized IR persisted under ``03_primary``
- sequence projection persisted under the partitioned ``04_feature`` contract
- vocabulary-backed tokenized document rows persisted under ``05_model_input``
- decoder-only Transformer checkpoints persisted under ``06_models``
- decoded evaluation samples persisted under ``07_model_output``
- metrics, Markdown summaries, and run metadata persisted under ``08_reporting``

Execution shards are persistence and scheduling units only. Experiment splits remain
score-level assignments derived from the split manifest.

Version-Key Derivation
----------------------

MotifML derives training contract versions through canonical JSON hashing in
``motifml.training.versioning``. Hash inputs are normalized before hashing:

- mappings are sorted by stringified key
- sets are sorted after canonicalization
- dataclasses, enums, and paths are converted into JSON-stable forms

The baseline version keys follow these dependency rules:

``normalized_ir_version``
   hashes the frozen normalized-IR contract payload plus normalization rules.

``feature_version``
   hashes ``normalized_ir_version`` together with the feature projection config and the
   frozen ``sequence_schema_version``.

``split_version``
   hashes the sorted corpus membership, the split configuration, and the split seed.

``vocabulary_version``
   hashes ``feature_version`` together with tokenization config, ``split_version``, the
   split seed, and the special-token policy.

``model_input_version``
   hashes ``feature_version`` together with ``vocabulary_version``, the model-input
   config, the special-token policy, and the storage schema version.

These rules are part of the persisted contract. If one of the frozen inputs changes, the
derived version key should change in the downstream artifact family that depends on it.

Configuration Families
----------------------

The baseline training stack is configured through these top-level Kedro parameter
families:

``data_split``
   score-level ratios, grouping behavior, and split hash seed.

``sequence_schema``
   the baseline token-emission schema, including note payload fields, structure
   markers, and control-family inclusion.

``vocabulary``
   special-token strings, frequency/size limits, and vocabulary guardrails.

``model_input``
   context length, stride, padding behavior, special-token policy, storage backend, and
   model-input reporting thresholds.

``model``
   baseline decoder-only Transformer architecture settings.

``training``
   device, optimizer, scheduler, batch size, gradient clipping, and epoch count.

``evaluation``
   evaluation splits, decoding limits, qualitative sample extraction, and evaluation
   guardrails.

``seed``
   the shared deterministic experiment seed reused by training and evaluation helpers.

Token and Split Semantics
-------------------------

The baseline path consumes the explicit ``baseline_v1`` sequence schema. That contract
freezes:

- note payload fields
- structure marker inclusion
- supported point and span control kinds
- special-token strings and policy semantics for BOS, EOS, padding, and ``<unk>``

Split assignment is score-level and deterministic. Documents are grouped by
``document_id`` by default, with ``relative_path`` as the configured fallback. Windowing
must happen strictly after split assignment so train, validation, and test windows never
cross split boundaries.

Persisted Layout
----------------

The implemented baseline does not treat ``04_feature`` or ``05_model_input`` as single
monolithic files. Both stages are partitioned, metadata-backed contracts.

The main training and evaluation surfaces are:

.. code-block:: text

   data/04_feature/ir/
     parameters.json
     records/**/*.feature.json

   data/05_model_input/ir/
     parameters.json
     vocabulary.json
     vocabulary_version.json
     model_input_version.json
     storage_schema.json
     records/<split>/<shard_id>/**/*.model_input.parquet

   data/06_models/training/baseline/
     checkpoint_manifest.json
     best_checkpoint.json
     checkpoints/*.pt
     model_config.json
     training_config.json
     run_metadata.json

   data/07_model_output/evaluation/
     qualitative_samples.json

   data/08_reporting/training/
     split_stats.json
     vocab_stats.json
     model_input_stats.json
     model_input_report.md
     training_history.json
     training_run_metadata.json
     metrics.json
     qualitative_report.md
     evaluation_run_metadata.json

Lazy Runtime and Binary Dataset Expectations
--------------------------------------------

``05_model_input`` is frozen as a Parquet-backed binary dataset contract. Training and
evaluation should consume it through:

- ``motifml.datasets.tokenized_model_input_runtime_dataset.TokenizedModelInputRuntimeDataset``
- the lazy runtime helpers under ``motifml.training``

Treat corpus-wide materialization of tokenized rows or reconstructed windows as a
regression. The approved baseline path expects lazy split-aware iteration over the
persisted binary-backed contract rather than notebook-local or ad hoc in-memory dataset
surfaces.
