Normalized IR Contract
======================

This page documents the implemented ``03_primary`` contract between canonical IR build
and downstream training preparation.

Role of ``03_primary``
----------------------

``03_primary`` is the last persisted stage whose artifacts still represent music-domain
truth rather than model-preparation artifacts.

The current normalization pipeline is still a deterministic passthrough, but its contract
is now explicit:

- one normalized IR document per source score
- stable source-relative identity carried forward from upstream IR persistence
- no model-specific flattening, token IDs, split labels, or window metadata
- explicit ``normalized_ir_version`` metadata persisted beside the normalized corpus

Task-Agnostic Guarantees
------------------------

The normalization parameter surface freezes these guarantees:

- ``stable_source_relative_identity``
- ``task_agnostic_domain_truth``
- ``no_model_specific_flattening``
- ``no_model_specific_windowing``

Those guarantees are recorded in the ``normalized_ir_version`` metadata artifact together
with the upstream IR schema version, serialized document format, and normalization
strategy.

Contract Enforcement
--------------------

Before ``normalized_ir_version`` is emitted, MotifML now validates that normalized IR
documents have not leaked training-facing fields.

The default forbidden field set includes:

- ``token_ids``
- ``window_start_offsets``
- ``split`` / ``split_version``
- ``attention_mask``
- ``input_ids`` / ``target_ids``
- ``vocabulary_version``
- ``model_input_version``

The exact list is configuration-backed under
``params:normalization.forbidden_model_fields`` so contract reviews stay explicit.

Persisted Metadata
------------------

The normalization stage now emits:

- normalized IR documents under ``data/03_primary/ir/documents/``
- ``normalized_ir_version.json`` under ``data/03_primary/ir/``
- shard-local normalized-version fragments under ``data/03_primary/ir/versions/shards/``

Shard-local version fragments must agree exactly; reducer wiring fails fast if different
shards attempt to publish incompatible normalized-IR contracts.
