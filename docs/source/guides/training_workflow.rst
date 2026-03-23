Training Workflow
=================

This guide describes the approved maintainer workflow for MotifML's baseline training
and evaluation stack. Use it when you need to rerun the baseline, inspect runtime
artifacts, or intentionally regenerate the tracked training fixtures.

The authoritative recovery baseline is now the V1 continuation task documented in
:doc:`../reference/v1_continuation_task_contract`. The default preprocessing flow
persists a dedicated continuation-example dataset under ``05_model_input``. The older
document-row tokenization path still exists as a legacy comparison/runtime surface until
the training and evaluation stacks are rewritten around continuation examples.

Canonical Run Paths
-------------------

MotifML keeps preprocessing, training, and evaluation as named Kedro pipelines so
maintainers can choose the scope they need:

.. code-block:: bash

   uv run kedro run --async
   uv run kedro run --pipelines=continuation_dataset
   uv run kedro run --pipelines=baseline_training
   uv run kedro run --pipelines=evaluation
   uv run kedro run --pipelines=baseline_training_evaluation

Use these command paths intentionally:

- ``__default__`` via ``uv run kedro run --async`` refreshes the baseline training-prep
  surface, including the V1 continuation dataset under ``05_model_input`` plus the
  legacy document-row artifacts still used by the current sequence-model runtime.
- ``continuation_dataset`` rebuilds just the deterministic V1 prompt/scaffold/target
  extraction surface from normalized IR plus the split manifest.
- ``baseline_training`` runs the default preprocessing path plus baseline training and
  persists ``06_models`` plus training reporting under ``08_reporting/training``.
- ``evaluation`` reuses an existing best checkpoint and persisted ``05_model_input`` to
  produce decoded outputs and evaluation reporting.
- ``baseline_training_evaluation`` is the canonical single-command review path from
  raw corpus inputs through evaluation outputs.

Configuration and Overrides
---------------------------

Baseline behavior is frozen through the training-phase parameter families in
``conf/base/parameters.yml``:

- ``data_split`` controls score-level split assignment and the split hash seed.
- ``continuation_dataset`` freezes the V1 structural eligibility and prompt-window
  contract used when extracting continuation examples from ``03_primary`` normalized IR.
- ``sequence_schema`` freezes the baseline token-emission surface from ``04_feature``.
- ``vocabulary`` controls special tokens, size limits, and acceptance guardrails.
- ``model_input`` controls context length, stride, padding, special-token policy, and
  the binary storage backend.
- ``model`` controls the decoder-only Transformer architecture.
- ``training`` controls device, optimizer, scheduler, batch size, and epoch count.
- ``evaluation`` controls evaluation splits, decoding limits, qualitative samples, and
  guardrail thresholds.
- ``seed`` freezes the shared deterministic experiment seed.

Experiments should be expressed through Kedro overrides rather than code edits. For
example:

.. code-block:: bash

   uv run kedro run --pipelines=baseline_training_evaluation \
     --params training.device=cuda,training.num_epochs=20,evaluation.qualitative.samples_per_split=8

Artifact Review Surfaces
------------------------

The baseline run produces five primary artifact families:

- ``data/05_model_input/v1_continuation/`` for persisted prompt/scaffold/target
  continuation examples plus shared dataset parameters
- ``data/05_model_input/ir/`` for the legacy tokenized-document rows, vocabulary
  metadata, version keys, and storage-schema metadata
- ``data/06_models/training/baseline/`` for checkpoints plus frozen model and training
  configs
- ``data/07_model_output/evaluation/`` for decoded qualitative sample payloads
- ``data/08_reporting/training/`` for split stats, continuation-dataset reporting,
  vocabulary/model-input reports, training history, metrics, Markdown summaries, and
  run metadata

The section 15 notebooks are the supported interactive inspection surfaces for these
artifacts:

- ``notebooks/model_input_inspection.ipynb``
- ``notebooks/tokenization_validation.ipynb``
- ``notebooks/training_run_review.ipynb``
- ``notebooks/training_failure_analysis.ipynb``

By default they look for runtime outputs under ``data/``. For ad hoc review of a
temporary test run, set ``MOTIFML_TRAINING_ARTIFACT_ROOT`` to the artifact directory
that contains the temporary ``model_input/``, reporting files, and evaluation outputs.
``model_input_inspection.ipynb`` also honors ``MOTIFML_MODEL_INPUT_DOCUMENT``, and
``tokenization_validation.ipynb`` honors ``MOTIFML_TOKENIZATION_DOCUMENT`` for
document-specific traces.

Lazy Loading Expectations
-------------------------

Lazy ``05_model_input`` consumption is a hard contract, not an optimization detail.
Training and evaluation must stream tokenized rows and reconstructed windows through
``TokenizedModelInputRuntimeDataset`` and the helpers under ``motifml.training``.

Do not introduce code that:

- materializes every tokenized document row in memory before iteration
- reconstructs every training window for the whole corpus up front
- bypasses the persisted ``05_model_input`` contract with notebook-local or script-local
  loaders in production code

Regression coverage should continue to prove that the first batch can be assembled
without touching later documents in the corpus.

Regenerating Tracked Training Artifacts
---------------------------------------

The tracked training fixture slice and normalized smoke bundle live under
``tests/fixtures/training/``. They are managed artifacts and should only change when an
intentional contract or behavior change is being reviewed.

Regenerate them with:

.. code-block:: bash

   uv run python tools/regenerate_training_fixtures.py

Run that command when a change intentionally affects:

- split planning behavior
- continuation-example extraction, rejection logic, or continuation-dataset version keys
- vocabulary contents or version-key derivation
- tokenized row structure or model-input metadata
- baseline training metadata, metrics, or qualitative evaluation outputs

Review the resulting diffs directly. The tracked fixture bundle is part of the
repository's regression surface, not just local developer convenience data.
