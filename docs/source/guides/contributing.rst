Contributing to MotifML
=======================

This is the project-wide contributor guide for MotifML. Use it as the default guidance
for repository work across pipelines, datasets, modeling code, evaluation code, and
documentation. Subsystem-specific guidance, including IR-specific rules, should be read
as supplements to this page rather than as the whole project process.

Project Posture
---------------

MotifML is a research codebase, but it should be maintained like serious production
infrastructure:

- repository behavior should be explicit, reproducible, and reviewable
- pipeline contracts should be documented and configuration-driven
- data transformations should remain deterministic for identical inputs
- experiments should be expressible through Kedro configuration rather than ad hoc code
  edits

The current implemented scope is strongest on symbolic data engineering and IR
construction. The documentation structure is intentionally broader so future training,
evaluation, and analysis systems can be documented without reorganizing everything again.

Core Engineering Expectations
-----------------------------

Kedro remains the orchestration layer for production pipeline code.

- keep transformation logic in ``src/motifml/``
- route pipeline data through the Kedro Data Catalog
- avoid direct file I/O inside Kedro nodes
- keep nodes small, typed, and side-effect free
- prefer multiple focused pipelines over one monolithic DAG

Configuration should own variable behavior.

- expose tunable behavior in ``conf/base/`` or environment-specific overrides
- avoid hardcoding experiment constants, device choices, or preprocessing assumptions
- document new parameters, catalog entries, or artifact formats when they are added

Reproducibility is a project-wide requirement.

- identical code, configuration, and inputs should produce stable pipeline outputs
- generated artifacts committed to git should be reproducible from tracked tools
- any change that affects data contracts or training behavior should be visible in config
  or in a documented artifact contract

Where Different Kinds of Work Belong
------------------------------------

``src/motifml/``
   Production library code, Kedro datasets, pipeline nodes, and reusable ML components.

``conf/``
   Shared configuration, catalog wiring, parameters, and logging.

``tests/``
   Unit tests, integration tests, and fixture-backed regression coverage.

``docs/source/``
   Project documentation. Keep it aligned with implemented behavior.

``notebooks/``
   Exploration, inspection, and visualization. Production logic should graduate out of
   notebooks and into ``src/motifml/``.

Making Changes
--------------

When changing the codebase:

- keep public interfaces typed and documented
- preserve clear boundaries between datasets, pipeline orchestration, and domain logic
- update schemas, serializers, parameters, and docs together when contracts change
- keep generated and tracked artifacts intentional; do not hand-edit generated outputs
  when a repository tool is responsible for producing them

If a change affects persisted datasets, schema surfaces, or pipeline outputs, call that
out explicitly in the PR description so the impact is easy to trace later.

Lazy ``05_model_input`` loading is a hard regression surface.

- training and evaluation code must consume tokenized documents and token windows
  through the lazy runtime handle or equivalent streaming iterators
- do not introduce corpus-wide in-memory materialization of tokenized rows or training
  windows as part of the normal baseline path
- keep regression tests that prove the first batch can be assembled without touching
  later documents in the corpus

Testing and Verification
------------------------

Use the smallest high-signal verification set that proves the change:

- targeted unit tests for local logic changes
- pipeline or integration tests for changes to orchestration or data contracts
- fixture and artifact regeneration when tracked outputs legitimately change
- documentation builds when the documentation structure or references change

Project-wide checks commonly include:

.. code-block:: bash

   uv run ruff check .
   uv run ruff format .
   uv run mypy src
   uv run pytest

Run narrower checks when they provide faster signal, but keep the final change set
adequately verified for its scope.

Tracked training fixtures and the tiny smoke bundle can be regenerated intentionally
with:

.. code-block:: bash

   uv run python tools/regenerate_training_fixtures.py

Documentation Expectations
--------------------------

MotifML's docs are organized into three buckets:

- ``overview/`` for system-level orientation
- ``guides/`` for contributor-facing workflow and engineering guidance
- ``reference/`` for technical contracts and subsystem interfaces

When implemented behavior changes, update the relevant docs in the same change set. The
goal is for a contributor to understand the current repository from the checked-in docs,
not from tribal memory or temporary notes.

Specialized Guides
------------------

Use these pages when working in specific parts of the project:

- :doc:`/overview/architecture` for the current pipeline and repository layout
- :doc:`/guides/ir_engineering` for IR-specific implementation guidance
- :doc:`/guides/inspection_artifacts` for tracked fixture, golden, and inspection
  surfaces
- :doc:`/reference/ir_contract` for the current canonical IR contract

As MotifML grows beyond the current IR-heavy stage, new subsystem guides should live
alongside these pages rather than forcing contributors to reinterpret IR-specific docs as
project-wide rules.
