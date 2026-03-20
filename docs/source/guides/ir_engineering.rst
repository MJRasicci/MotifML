IR Engineering
==============

This guide supplements :doc:`/guides/contributing` for work that changes MotifML's IR
implementation. It focuses on the practical engineering rules for extending or modifying
the IR surface without treating the IR as the whole project.

Determinism Requirements
------------------------

IR work must remain reproducible and byte-stable for identical inputs.

The current determinism layers are:

- ``src/motifml/ir/ids.py`` defines identifier builders and canonical sort keys
- ``src/motifml/pipelines/ir_build/models.py`` normalizes ordering inside typed emission
  results
- ``src/motifml/ir/serialization.py`` canonicalizes collections again before JSON output
- ``src/motifml/datasets/motif_ir_corpus_dataset.py`` skips rewriting byte-identical
  documents

When modifying IR data or build logic:

- derive identifiers from stable source position, never from unordered traversal
- reuse the existing sort-key helpers instead of introducing ad hoc ordering rules
- keep configuration-driven behavior in Kedro parameters rather than in node logic
- add or update tests that prove the new behavior remains deterministic

Tracked Regression Surfaces
---------------------------

The repository uses tracked fixtures and generated inspection artifacts to keep the IR
surface stable over time.

Key inputs and outputs are:

- ``tests/fixtures/motif_json/`` for curated raw Motif JSON fixtures
- ``tests/fixtures/ir/golden/`` for a tracked golden subset of persisted IR documents
- ``tests/fixtures/ir/review_bundles/`` for deterministic tabular and SVG inspection
  bundles
- ``tests/fixtures/ir_fixture_catalog.json`` for fixture metadata and coverage mapping

Regenerate those artifacts with the project tools rather than by editing generated files
manually:

.. code-block:: bash

   uv run python tools/regenerate_ir_fixture_corpus.py
   uv run python tools/generate_ir_review_bundles.py

The generator scripts own the generated fixture JSON, golden IR, and inspection-bundle
artifacts under ``tests/fixtures/``.

Extending IR Entity Families
----------------------------

When you add a new IR entity field, control kind, or technique payload, update the full
typed surface instead of changing only one layer.

For most IR-surface changes that means touching:

- the dataclass model in ``src/motifml/ir/models.py``
- serialization and deserialization in ``src/motifml/ir/serialization.py``
- any relevant build-node coercion logic under ``src/motifml/pipelines/ir_build/``
- the JSON schema in
  ``src/motifml/ir/schema/motifml-ir-document.schema.json``
- validation, reporting, inspection-table, or visualization code if the new surface
  should be testable or inspectable

Adding a Point or Span Control Kind
-----------------------------------

For a new control family, use this sequence:

#. extend the relevant enum and typed payload model in ``src/motifml/ir/models.py``
#. update payload validation in the corresponding control-value validator
#. add serializer and deserializer support in ``src/motifml/ir/serialization.py``
#. wire the raw Motif JSON mapping in the ``ir_build`` control-node logic
#. update the JSON schema
#. add unit tests and fixture-backed coverage
#. update inspection tables or visualizations if the new control needs to be visible in
   deterministic artifact outputs

Adding Technique Support
------------------------

Technique support is intentionally split across onset-local and note-local surfaces.

Current mapping entry points include:

- onset-local techniques in the onset emission logic
- note-local techniques in the note emission logic
- typed payloads in ``GenericTechniqueFlags``, ``GeneralTechniquePayload``,
  ``StringFrettedTechniquePayload``, and ``TechniquePayload``

When adding a technique field:

#. decide whether it belongs in the generic, general, or string-fretted payload
#. update the typed model and serializer / deserializer
#. teach the relevant build node how to populate it from raw Motif JSON
#. expose it in inspection artifacts if developers need to see it during regression
#. add both unit-level and fixture-backed coverage

Unsupported Feature Reporting
-----------------------------

Unsupported or intentionally excluded source features must remain observable.

Current expectations are:

- emit ``IrBuildDiagnostic`` entries instead of silently dropping data
- let ``build_ir_manifest`` group diagnostics into manifest-level summaries
- preserve unsupported or excluded feature codes in the manifest summary
- keep corpus-level aggregation working through
  ``src/motifml/pipelines/ir_validation/nodes.py``

Use diagnostic categories consistently:

- ``unsupported`` for source features the IR does not yet represent
- ``excluded`` for features the pipeline intentionally skips in the current contract
- ``malformed`` for invalid or structurally unusable source content
- ``other`` for diagnostics that do not fit the categories above

Recommended Verification
------------------------

For IR changes, the highest-signal checks are:

.. code-block:: bash

   uv run pytest tests/ir/test_serialization.py
   uv run pytest tests/ir/test_fixture_corpus.py
   uv run pytest tests/pipelines/test_ir_build_fixture_determinism.py
   uv run pytest tests/ir/test_review_bundles.py
   uv run pytest tests/pipelines/test_ir_pipeline_integration.py

Run additional targeted tests for the specific pipeline nodes, projection modules, or
validators you changed.
