# MotifML Agent Instructions

MotifML is a structured ML pipeline for modeling symbolic musical data. It transforms music representations into datasets for training neural networks to predict, generate, and analyze musical structure.

Agents assist with implementation, refactoring, experimentation, and documentation. **Defer to the human maintainer on:** architecture decisions, domain modeling, pipeline structure, dataset definitions, and training methodology.

---

## Core Engineering Principles

### Reproducibility

- All pipelines must be deterministic: explicit configuration, versioned parameters, no hidden state, no randomness without a fixed seed
- `kedro run` on identical code + parameters + data must reproduce training results
- Any change affecting training behavior must surface in config/params

### Pipeline Architecture (Kedro)

MotifML uses **Kedro** for all orchestration. Agents must not bypass Kedro to implement data processing or training logic.

Pipeline stages:

```
01_raw → 02_intermediate → 03_primary → 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting
```

Node rules:
- One logical transformation per node
- Pure functions: typed inputs/outputs, no side effects, no global state
- **No direct file I/O inside nodes** — all data flows through the Kedro Data Catalog
- No hardcoded paths (e.g., `open("data/01_raw/...")`); reference datasets by catalog name

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
    normalized_tracks: list[Track],
    parameters: FeatureParameters
) -> FeatureDataset:
    """Extract musical features from normalized tracks."""
```

Keep functions small; break complex logic into helpers.

### Pipeline Composition

Prefer multiple small, composable pipelines over one monolithic pipeline:

```
kedro run --pipeline=feature_extraction
kedro run --pipeline=training
kedro run
```

Pipelines must not tightly couple unrelated stages.

### Data Stage Responsibilities

| Stage | Purpose | Rules |
|---|---|---|
| `01_raw` | External/imported source data | **Never modify** |
| `02_intermediate` | Temporary transform artifacts | Not canonical |
| `03_primary` | Cleaned, normalized domain model data | Canonical for feature extraction |
| `04_feature` | Engineered features | — |
| `05_model_input` | Tensors, windows, tokenized sequences | Document assumptions (resolution, length, tokenization) |
| `06_models` | Serialized models and checkpoints | — |
| `07_model_output` | Predictions, generated sequences | — |
| `08_reporting` | Metrics, plots, summaries | All metrics written here |

---

## ML & Training Guidelines

### Code Separation

```
model/       # architecture definitions
training/    # training loops
evaluation/  # metrics and analysis
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

## Agent Safety Rules

Agents must **not**:
- Delete or alter raw data
- Change pipeline structure without explanation
- Introduce hidden side effects
- Bypass the Kedro Data Catalog
- Commit large data artifacts

If a task requires any of the above, **propose it for human review** rather than implementing it.

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

If a task involves altering the domain model, changing pipeline structure, modifying dataset formats, or introducing new ML architectures — **propose a design first**. Architectural integrity takes priority over implementation speed.
