IR Review Workflow
==================

Use this checklist for pull requests that change IR mapping, serialization, validation,
fixtures, or review artifacts.

Review Checklist
----------------

- confirm determinism still holds:
  repeated runs, reordered raw-document inputs, and canonical serialization should stay
  byte-stable
- confirm forbidden metadata is still excluded from persisted IR documents
- confirm fixture and golden-artifact updates were regenerated with the tracked tools
  instead of hand-edited
- confirm validation coverage exists for every new mapping branch, payload shape, or
  unsupported-feature path
- confirm unsupported or excluded source features remain visible in the manifest,
  validation summary, or review bundles
- confirm at least one reviewed golden artifact change is included whenever mapping logic
  changes the persisted IR shape

Golden Artifact Review Requirement
----------------------------------

For any change that alters persisted IR shape, machine-only verification is not enough.
At least one changed tracked artifact should be inspected during review before the work
is treated as approved.

Acceptable review surfaces include:

- a changed file under ``tests/fixtures/ir/golden/``
- a changed bundle under ``tests/fixtures/ir/review_bundles/``
- the corresponding entry in ``tests/fixtures/ir_fixture_catalog.json`` when review status
  changes

Recommended Review Sequence
---------------------------

#. regenerate the tracked fixture corpus:

   .. code-block:: bash

      uv run python tools/regenerate_ir_fixture_corpus.py

#. regenerate review bundles for the affected fixtures:

   .. code-block:: bash

      uv run python tools/generate_ir_review_bundles.py --fixture-id <fixture_id>

#. inspect the changed IR JSON, bundle README, validation report, CSV tables, and SVG
   visualizations
#. keep new artifacts at ``pending_review`` until that inspection is complete
#. only mark an artifact ``approved`` when the reviewed output accurately represents the
   intended persisted IR shape

When No Golden Artifact Changes
-------------------------------

If a PR touches IR code but does not change persisted IR shape, say so explicitly in the
review summary. Reviewers should not have to infer that from a missing fixture diff.

Current Enforcement Points
--------------------------

The repository already encodes part of this workflow in tests:

- ``tests/ir/test_fixture_corpus.py`` guards fixture coverage, schema validity, and the
  default pending-review state for newly generated goldens
- ``tests/pipelines/test_ir_build_fixture_determinism.py`` checks byte-stable IR and
  manifest output
- ``tests/ir/test_review_bundles.py`` requires checked-in review bundles to reproduce
  exactly

The PR template mirrors this checklist so reviewers can apply the same rules during code
review.
