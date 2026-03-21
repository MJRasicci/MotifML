Model Input Storage Contract
============================

This page freezes the v1 binary storage decision for tokenized ``05_model_input``
artifacts. The Parquet-backed dataset implementation now lives in
``motifml.datasets.tokenized_model_input_dataset.TokenizedModelInputDataset``, and the
tokenization pipeline now persists that contract end to end. Downstream work targets one
stable backend, schema version, and physical layout.

Backend Choice
--------------

MotifML uses ``parquet`` as the v1 binary backend for dense tokenized document payloads.
The backend choice is surfaced through ``model_input.storage.backend`` and validated
through ``motifml.datasets.model_input_storage.ModelInputStorageSchema``.

Frozen Schema Version
---------------------

The v1 storage schema version is ``parquet-v1``. That value is carried in:

- ``conf/base/parameters.yml`` under ``model_input.storage.schema_version``
- persisted model-input metadata via ``storage_schema_version``
- the standalone ``storage_schema.json`` metadata surface persisted alongside Parquet
  rows

Physical Layout
---------------

The tokenized document dataset is partitioned by experiment split and execution shard.
One logical row is persisted per source document.

For the default non-sharded pipeline, MotifML persists rows under a synthetic
``global`` shard partition. Shard-local execution preserves the concrete shard id from
``params:execution.shard_id`` so partitioned and non-partitioned runs share one stable
layout and loader contract.

The frozen on-disk layout is:

.. code-block:: text

   data/05_model_input/ir/
     parameters.json
     vocabulary.json
     vocabulary_version.json
     model_input_version.json
     storage_schema.json
     records/
       <split>/
         <shard_id>/
           <source-relative-path>.model_input.parquet

Examples:

.. code-block:: text

   data/05_model_input/ir/records/train/global/fixtures/example.json.model_input.parquet
   data/05_model_input/ir/records/train/shard-00003/collection/demo.json.model_input.parquet
   data/05_model_input/ir/records/validation/shard-00012/fixtures/example.json.model_input.parquet

Contract Helpers
----------------

``src/motifml/datasets/model_input_storage.py`` owns the frozen storage constants and the
canonical record-path builder:

- backend: ``parquet``
- storage schema version: ``parquet-v1``
- record suffix: ``.model_input.parquet``
- partition fields: ``split``, ``shard_id``

Those helpers remain separate from the Kedro dataset class so record models, dataset
persistence, and reporting can all reuse one storage-layout contract.

Lazy Runtime Consumption
------------------------

The persisted ``05_model_input`` contract is designed to be consumed lazily at runtime.
Training and evaluation should stream tokenized documents and token windows through
``motifml.datasets.tokenized_model_input_runtime_dataset.TokenizedModelInputRuntimeDataset``
and the ``motifml.training`` lazy loader helpers rather than building corpus-wide
in-memory lists.

Treat corpus-wide materialization of tokenized document rows or reconstructed training
windows as a regression against the baseline training contract. Maintainers should keep
tests that prove the first batch can be assembled without touching later documents in
the corpus.
