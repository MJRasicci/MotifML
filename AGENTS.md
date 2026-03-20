# MotifML Agent Instructions

MotifML is a Kedro-based symbolic-music research codebase. The repository currently
focuses on raw-corpus ingestion, canonical IR construction and validation, reporting, and
baseline normalization, feature-extraction, and tokenization stages that prepare
downstream ML-ready datasets.

Training, generation, and evaluation systems are planned project directions, but they are
not yet implemented as first-class pipelines in this repository.

Agents assist with implementation, refactoring, experimentation, and documentation.
**Defer to the maintainer on:** architecture decisions, domain modeling, pipeline
structure, dataset definitions, and training methodology. If a task touches those areas,
pause and propose a design rather than deciding unilaterally.

---

## Core Engineering Principles

### Reproducibility

- All pipelines must be deterministic: explicit configuration, versioned parameters, no hidden state, no randomness without a fixed seed
- `kedro run` on identical code + parameters + data must reproduce pipeline outputs and, where applicable, training results
- Tracked generated artifacts must be reproducible from repository tools
- Any change affecting training behavior or persisted data contracts must surface in config/params

### Pipeline Architecture (Kedro)

MotifML uses **Kedro** for all orchestration. Agents must not bypass Kedro to implement data processing or training logic.

Pipeline stages:

```
00_corpus → 01_raw → 02_intermediate → 03_primary → 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting
```

Node rules:
- One logical transformation per node
- Pure functions: typed inputs/outputs, no side effects, no global state
- **No direct file I/O inside nodes** — all data flows through the Kedro Data Catalog
- No hardcoded paths (e.g., `open("data/01_raw/...")`); reference datasets by catalog name

Current implemented pipelines:

- `ingestion`: builds and summarizes the raw Motif JSON corpus
- `ir_build`: validates raw Motif JSON and emits canonical IR documents plus a manifest
- `ir_validation`: produces IR validation reports and corpus summaries
- `normalization`: current deterministic passthrough baseline from canonical IR to normalized IR
- `feature_extraction`: projects normalized IR into sequence, graph, or hierarchical views
- `tokenization`: packages extracted features into baseline model-input artifacts

Do not assume training, generation, or evaluation pipelines already exist unless they are
explicitly added to the repository.

### Domain Model Integrity

- Raw format adapters (Guitar Pro, MIDI, MusicXML, etc.) convert into the domain model
- ML pipelines operate on normalized domain representations
- Format-specific metadata must not pollute core musical abstractions
- Extend transformation layers; do not modify the core domain model without explicit direction

### Configuration over Hardcoding

All variable behavior (model architecture, hyperparameters, dataset splits, device selection, preprocessing) must live in `conf/base/` or Kedro parameter files. Never hardcode experimental constants.

---

## Code Quality Standards

| Area | Requirements |
|---|---|
| **Readability** | Explicit, well-named, easy to review/refactor. No clever one-liners. |
| **Types** | Static typing on all public functions and important APIs; avoid untyped containers where structure matters |
| **Docstrings** | All public functions and pipeline nodes: purpose, inputs, outputs, assumptions, side effects |
| **Testing** | Cover data transformation correctness, dataset integrity, deterministic behavior; no large external datasets |

---

## Kedro-Specific Rules

### Node Signatures

```python
def extract_features(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    parameters: FeatureExtractionParameters,
) -> IrFeatureSet:
    """Project normalized IR documents into the configured feature surface."""
```

Keep functions small; break complex logic into helpers.

### Pipeline Composition

Prefer multiple small, composable pipelines over one monolithic pipeline:

```
kedro run --pipeline=ingestion
kedro run --pipeline=feature_extraction
kedro run
```

Pipelines must not tightly couple unrelated stages.

### Data Stage Responsibilities

| Stage | Purpose | Rules |
|---|---|---|
| `00_corpus` | External source corpus | **Never modify**; this is the imported source boundary |
| `01_raw` | Generated raw Motif JSON corpus | Derived from `00_corpus`; do not hand-edit |
| `02_intermediate` | Transform artifacts, manifests, and canonical IR outputs | Not the normalized primary layer |
| `03_primary` | Cleaned, normalized domain model data | Current baseline stores normalized IR here |
| `04_feature` | Engineered features | — |
| `05_model_input` | Tensors, windows, tokenized sequences | Document assumptions (resolution, length, tokenization) |
| `06_models` | Serialized models and checkpoints | — |
| `07_model_output` | Predictions, generated sequences | — |
| `08_reporting` | Metrics, plots, summaries | All metrics written here |

---

## ML & Training Guidelines

These guidelines apply when training and evaluation systems are introduced. They are
intended to shape future work, not to imply that those subsystems are already present.

### Code Separation

```
src/motifml/model/       # architecture definitions
src/motifml/training/    # training loops
src/motifml/evaluation/  # metrics and analysis
```

Never write monolithic scripts that combine dataset loading, training, and evaluation.

### PyTorch

- Explicit tensor shapes, deterministic seeding, separated model/training logic
- Support both CPU and CUDA; device selection must be configuration-driven

### Dataset Construction

- Splits, windowing, padding/truncation must all be deterministic and explicitly documented
- Document assumptions: time resolution, sequence length, tokenization strategy

### Experiments

Experiments should require no code changes — driven by param files and config overrides:

```
kedro run --params training.learning_rate=0.0005
```

### Notebooks
Allowed for exploration, inspection, and visualization only. Production logic lives in `src/motifml/`.

---

## Generated Artifacts and Documentation

- Keep documentation in `docs/source/overview/`, `docs/source/guides/`, and `docs/source/reference/` aligned with implemented behavior
- Do not hand-edit generated fixture or inspection artifacts when a repository tool owns them
- Use the tracked generators for managed artifacts:
  - `uv run python tools/regenerate_ir_fixture_corpus.py`
  - `uv run python tools/generate_ir_inspection_bundles.py`
- If a change affects persisted contracts, fixture outputs, or tracked inspection artifacts, update the docs and regenerate the managed artifacts in the same change set

Managed generated surfaces include:

- `tests/fixtures/ir/golden/`
- `tests/fixtures/ir/inspection_bundles/`
- `tests/fixtures/ir_fixture_catalog.json`

---

## Agent Safety Rules

Agents must **not**:
- Delete or alter raw data
- Change pipeline structure without explanation
- Introduce hidden side effects
- Bypass the Kedro Data Catalog
- Commit large data artifacts

If a task requires any of the above, pause and ask for explicit maintainer direction
rather than implementing it unilaterally.

---

## Commit Discipline

- Logically scoped, minimal, well-explained commits
- Messages describe **intent**, not just code changes

```
# Good
Add feature extraction node for rhythmic density metrics

# Bad
update code
```

---

## When in Doubt

If a task involves altering the domain model, changing pipeline structure, modifying
dataset formats, or introducing new ML architectures, pause and propose a design first.
Architectural integrity takes priority over implementation speed.
