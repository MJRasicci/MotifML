# MotifML

MotifML is a machine learning pipeline for modeling structured musical sequences across time, instruments, and expressive features.

The project focuses on transforming symbolic music formats into rich, learnable representations and training models capable of predicting, generating, and analyzing musical structure.

---

## Project Goals

MotifML is designed around three core principles:

### 1. Structured Musical Representation

* Encode polyphonic, multi-track music into consistent, model-friendly formats
* Preserve timing, articulation, dynamics, and harmonic relationships
* Support multiple input formats through a unified domain model

### 2. Reproducible ML Pipelines

* Use Kedro to define deterministic data and training pipelines
* Separate data processing, feature engineering, training, and evaluation
* Track all inputs and outputs via the Data Catalog

### 3. Extensibility

* Add new data sources (MIDI, MusicXML, etc.) without rewriting pipelines
* Swap model architectures without modifying upstream data flow
* Support both CPU and GPU execution environments

---

## Architecture Overview

MotifML follows a pipeline-oriented architecture:

```
Raw Data (GP / MIDI / MusicXML)
        ↓
Normalization / Parsing
        ↓
Feature Extraction
        ↓
Dataset Construction
        ↓
Model Training (PyTorch)
        ↓
Evaluation & Reporting
```

* **Kedro** orchestrates the pipeline and manages datasets
* **PyTorch** handles model definition and training
* **uv** provides reproducible environment and dependency management

---

## Project Structure

```
motifml/
├── src/motifml/           # Core library and pipeline code
├── conf/                  # Kedro configuration (catalog, parameters, logging)
├── data/                  # Data directories (ignored in git)
├── notebooks/             # Notebooks for data analysis and prototyping
├── tests/                 # Unit tests
├── pyproject.toml         # Project configuration
```

### Data Directory

The `data/` directory follows a staged structure:

```
data/
├── 01_raw/
├── 02_intermediate/
├── 03_primary/
├── 04_feature/
├── 05_model_input/
├── 06_models/
├── 07_model_output/
├── 08_reporting/
```

Data is **not versioned**. Only structure (`.gitkeep`) is committed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MJRasicci/MotifML.git
cd motifml
```

---

### 2. Create the environment

```bash
uv venv --python 3.11
uv sync --extra dev
```

---

### 3. Run the pipeline

```bash
uv run kedro run
```

---

### 4. Launch visualization tools

```bash
uv run kedro viz
uv run jupyter lab
```

---

## Development Workflow

### Linting & Formatting

```bash
uv run ruff check . --fix
uv run ruff format .
```

### Testing

```bash
uv run pytest
```

### Type Checking

```bash
uv run mypy src
```

### Pre-commit Hooks

Hooks run automatically on commit. To run manually:

```bash
uv run pre-commit run --all-files
```

---

## Device Configuration

MotifML supports both CPU and GPU execution.

Device selection should be controlled via configuration (not hardcoded), e.g.:

```yaml
training:
  device: auto
```

Where:

* `auto` → uses CUDA if available, otherwise CPU
* `cuda` → requires GPU
* `cpu` → forces CPU execution

---

## Current Status

MotifML is under active development. Initial focus areas:

* Core data model for symbolic music
* Feature extraction pipeline
* Baseline sequence models
* Dataset preparation tooling

---

## Future Work

* Multi-instrument harmonic modeling
* Temporal attention-based architectures
* Audio-aligned training (symbolic + waveform)
* Dataset expansion and augmentation strategies

---

## License

MIT — see `LICENSE.md`.

---

## Notes

* The datasets used for training are intentionally excluded from version control. For an example of a compatible dataset, see [PDMX](https://github.com/pnlong/PDMX).
* GPU support is optional and environment-dependent
* The project prioritizes correctness, structure, and reproducibility over rapid prototyping
