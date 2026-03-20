IR Contributor Guide
====================

This guide covers the concrete workflow for changing MotifML's IR implementation safely.
It focuses on the code that exists today rather than the broader design rationale.

Documentation Source of Truth
-----------------------------

Contributor-facing docs live in ``docs/source``. When architecture, IR behavior, or the
review workflow changes, update the corresponding ``.rst`` page in the same change set.

Canonical Ordering and Determinism
----------------------------------

IR work must remain byte-stable for identical inputs.

The current determinism layers are:

- ``src/motifml/ir/ids.py`` defines canonical id builders and sort-key helpers
- ``src/motifml/pipelines/ir_build/models.py`` sorts emitted collections inside typed
  result models
- ``src/motifml/ir/serialization.py`` re-canonicalizes document collections before JSON
  output
- ``src/motifml/datasets/motif_ir_corpus_dataset.py`` avoids rewriting byte-identical IR
  files

When you add or modify IR data:

- reuse the existing sort-key helpers instead of inventing ad hoc ordering
- keep ids derived from stable source position, never from unordered container traversal
- add or update determinism coverage in ``tests/ir/test_serialization.py`` and
  ``tests/pipelines/test_ir_build_fixture_determinism.py``

Golden Fixtures and Review Bundles
----------------------------------

The tracked fixture corpus is the review boundary for persisted IR changes.

Source-of-truth files:

- ``tests/fixtures/ir_fixture_catalog.json``: fixture inventory, coverage surface, schema
  paths, and golden-review status
- ``tests/fixtures/motif_json/README.md``: human-facing fixture description
- ``tests/fixtures/ir/golden/``: tracked golden IR subset
- ``tests/fixtures/ir/review_bundles/``: checked-in human review bundles

Use the repository tools instead of hand-editing generated artifacts:

.. code-block:: bash

   uv run python tools/regenerate_ir_fixture_corpus.py
   uv run python tools/generate_ir_review_bundles.py

Important workflow rules:

- the generator owns ``tests/fixtures/motif_json/``, ``tests/fixtures/ir/golden/``, and
  ``tests/fixtures/ir_fixture_catalog.json``
- newly generated golden IR artifacts default to
  ``provisional_pending_human_review``
- only a human reviewer should flip a golden artifact to ``approved_by_human``
- mapping changes that affect persisted IR shape should update both the relevant golden IR
  artifact and the checked-in review bundles

Adding a New Supported Control Kind
-----------------------------------

Adding a control kind touches more than one layer. Use this sequence to keep the contract
coherent:

#. extend the enum and typed payload model in ``src/motifml/ir/models.py``
#. update payload validation in ``_validate_point_control_value`` or
   ``_validate_span_control_value``
#. add deserialization support in ``src/motifml/ir/serialization.py``
#. wire raw Motif JSON mapping in ``src/motifml/pipelines/ir_build/nodes.py``:
   ``POINT_CONTROL_KIND_MAP`` or ``SPAN_CONTROL_KIND_MAP`` plus the value coercer
#. update the JSON schema in ``src/motifml/ir/schema/motifml-ir-document.schema.json``
#. add fixture-backed coverage and any review-table or visualization changes needed to
   make the new kind inspectable

Adding a Technique Payload Safely
---------------------------------

Technique support is split across onset-local and note-local surfaces.

Current mapping entry points:

- onset-local techniques: ``_coerce_optional_onset_techniques`` in
  ``src/motifml/pipelines/ir_build/nodes.py``
- note-local techniques: ``_coerce_optional_note_techniques`` in the same module
- typed payloads: ``GenericTechniqueFlags``, ``GeneralTechniquePayload``,
  ``StringFrettedTechniquePayload``, and ``TechniquePayload`` in
  ``src/motifml/ir/models.py``

For a new technique field:

#. decide whether it belongs on the generic, general, or string-fretted payload
#. update the typed model and serializer / deserializer
#. teach the raw-note or raw-beat coercer how to populate it
#. update review-table summaries if a human reviewer should be able to see it
#. add unit tests for model validation plus fixture-backed pipeline coverage

Unsupported Feature Reporting
-----------------------------

Unsupported or excluded source features must stay visible.

Current expectations:

- emit an ``IrBuildDiagnostic`` instead of silently dropping data
- let ``build_ir_manifest`` group diagnostics into ``conversion_diagnostics``
- ensure unsupported or excluded codes appear in
  ``IrManifestEntry.unsupported_features_dropped`` for quick inspection
- keep corpus summary aggregation working through
  ``src/motifml/pipelines/ir_validation/nodes.py``

In practice, use:

- ``unsupported`` for source features the IR does not currently represent
- ``excluded`` for intentionally skipped surfaces, such as open-ended spans that v1 does
  not persist
- ``malformed`` or ``other`` for canonical-surface problems that are not feature gaps

Recommended Verification
------------------------

For IR changes, the highest-signal checks are:

.. code-block:: bash

   uv run pytest tests/ir/test_serialization.py
   uv run pytest tests/ir/test_fixture_corpus.py
   uv run pytest tests/pipelines/test_ir_build_fixture_determinism.py
   uv run pytest tests/ir/test_review_bundles.py

Run more targeted pipeline-node tests as needed when you change specific mapping logic.
Use :doc:`ir_review_workflow` for the reviewer-facing checklist once the implementation is
ready for review.
