# MotifML

MotifML is a Kedro-based symbolic music pipeline for turning Motif exports into
deterministic, reviewable IR artifacts and downstream ML datasets.

The repository currently covers raw-corpus ingestion, canonical IR build and validation,
fixture-backed review tooling, and baseline normalization, feature-extraction, and
tokenization stages for downstream experiments.

## Current Scope

- ingest source corpora into canonical Motif JSON with deterministic manifests and
  summaries
- build canonical IR documents plus corpus manifests, validation reports, and scale
  summaries
- track regression surfaces through fixtures, golden IR artifacts, review bundles, and
  the IR inspection notebook
- project IR documents into sequence, graph, and hierarchical feature views
- package baseline model-input artifacts under Kedro

Training, generation, and evaluation pipelines are not implemented yet.

## Repository Layout

```text
src/motifml/           Core library and Kedro pipeline code
conf/                  Shared Kedro catalog, parameters, and logging config
docs/source/           Versioned architecture, IR, and workflow documentation
notebooks/             Inspection and exploration notebooks
tests/                 Unit, integration, and fixture-backed regression tests
tools/                 Fixture and review-bundle regeneration scripts
```

The data directory follows Kedro's staged layout:

```text
data/
├── 00_corpus/
├── 01_raw/
├── 02_intermediate/
├── 03_primary/
├── 04_feature/
├── 05_model_input/
├── 06_models/
├── 07_model_output/
└── 08_reporting/
```

Tracked source fixtures live under `tests/fixtures/`. Runtime corpora and model artifacts
under `data/` are intentionally excluded from version control.

## Getting Started

Create the environment:

```bash
uv venv --python 3.11
uv sync --extra dev
```

Add your source corpus anywhere under `data/00_corpus/` and place the
[Motif CLI](https://github.com/MJRasicci/Motif) binary at `tools/motif-cli`.

Build and summarize the raw Motif JSON corpus:

```bash
uv run kedro run --pipeline=ingestion
```

Run the full default pipeline:

```bash
uv run kedro run --async
```

Launch inspection tools when needed:

```bash
uv run kedro viz
uv run jupyter lab
```

## Fixture and Review Tooling

Regenerate the tracked raw fixtures and golden IR subset:

```bash
uv run python tools/regenerate_ir_fixture_corpus.py
```

Regenerate review bundles for the tracked review fixtures:

```bash
uv run python tools/generate_ir_review_bundles.py
```

The checked-in contributor workflow is documented in:

- `docs/source/architecture.rst`
- `docs/source/ir_design.rst`
- `docs/source/ir_contributor_guide.rst`
- `docs/source/ir_review_workflow.rst`

## Development

Run the main verification commands with:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run mypy src
uv run pytest
```

Run pre-commit hooks across the repository with:

```bash
uv run pre-commit run --all-files
```

## License

MIT; see `LICENSE.md`.
