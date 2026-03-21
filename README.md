# MotifML

MotifML is a Kedro-based symbolic-music research pipeline for turning Motif exports into
deterministic IR corpora and downstream ML-ready datasets.

The repository currently covers raw-corpus ingestion, canonical IR build and validation,
fixture-backed regression artifacts, deterministic split planning, and baseline
normalization, feature-extraction, tokenization, and baseline decoder-only Transformer
training stages for downstream experiments.

## Current Scope

- ingest source corpora into canonical Motif JSON with deterministic manifests and
  summaries
- build canonical IR documents plus corpus manifests, validation reports, and scale
  summaries
- track regression surfaces through fixtures, golden IR artifacts, inspection bundles,
  and the IR inspection notebook
- plan deterministic score-level experiment splits with persisted manifests and split
  summaries
- project IR documents into sequence, graph, and hierarchical feature views
- package baseline model-input artifacts under Kedro
- train the baseline decoder-only Transformer through Kedro-managed checkpoints and
  reporting

Generation and evaluation pipelines are not implemented yet.

## Repository Layout

```text
src/motifml/           Core library and Kedro pipeline code
conf/                  Shared Kedro catalog, parameters, and logging config
docs/source/           Versioned overview, guides, and technical reference docs
notebooks/             Inspection and exploration notebooks
tests/                 Unit, integration, and fixture-backed regression tests
tools/                 Fixture and inspection-bundle regeneration scripts
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

Run the full default preprocessing pipeline through tokenization:

```bash
uv run kedro run --async
```

Run the canonical single-command baseline training path:

```bash
uv run kedro run --pipeline=baseline_training
```

The default pipeline intentionally stops at `05_model_input`; heavy training work lives
behind the explicit `baseline_training` pipeline so maintainers can opt into it
deliberately.

Launch inspection tools when needed:

```bash
uv run kedro viz
uv run jupyter lab
```

## Fixtures and Inspection Artifacts

Regenerate the tracked raw fixtures and golden IR subset:

```bash
uv run python tools/regenerate_ir_fixture_corpus.py
```

Regenerate the tracked inspection bundles:

```bash
uv run python tools/generate_ir_inspection_bundles.py
```

Regenerate the tracked split-planning fixtures:

```bash
uv run python tools/regenerate_training_split_fixtures.py
```

The core project documentation is organized as:

- `docs/source/overview/`
- `docs/source/guides/`
- `docs/source/reference/`

Useful entry points include:

- `docs/source/index.rst`
- `docs/source/guides/contributing.rst`
- `docs/source/guides/ir_engineering.rst`
- `docs/source/reference/ir_contract.rst`

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
