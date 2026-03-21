Model Input Storage Contract
============================

This page freezes the v1 binary storage decision for tokenized ``05_model_input``
artifacts. The Parquet-backed dataset implementation now lives in
``motifml.datasets.tokenized_model_input_dataset.TokenizedModelInputDataset``, while the
tokenization pipeline wiring lands separately. Downstream work now targets one stable
backend, schema version, and physical layout.

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

The frozen on-disk layout is:

.. code-block:: text

   data/05_model_input/ir/
     parameters.json
     storage_schema.json
     records/
       <split>/
         <shard_id>/
           <source-relative-path>.model_input.parquet

Examples:

.. code-block:: text

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
