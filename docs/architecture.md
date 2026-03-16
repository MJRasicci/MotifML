# MotifML Architecture Overview

This document is the source of truth for MotifML's architecture and design decisions. It is a living document updated as the project evolves.

---

## System Overview

MotifML is an ML pipeline for modeling structured musical sequences. It ingests symbolic music data exported from [Motif](https://github.com/mjrasicci/Motif), transforms it into a trainable intermediate representation, and uses transformer-based models for both generative composition and targeted score editing.

```
Source Files (Guitar Pro, MusicXML, MIDI, ...)
        |
        v
   Motif (parse & export to JSON)
        |
        v
  01_raw: Motif JSON
        |
        v
  02_intermediate: Event/Graph IR
        |
        v
  03_primary: Normalized domain data
        |
        v
  04_feature: Engineered features
        |
        v
  05_model_input: Tokenized sequences / tensors
        |
        v
  06_models: Trained model checkpoints
        |
        v
  07_model_output: Generated / edited sequences
        |
        v
  08_reporting: Metrics, validation, analysis
```

---

## Data Flow

### Stage 1: Raw Ingestion (01_raw)

Source music files are processed through Motif, which exports a JSON representation of the score's domain model. This JSON captures the full structure of the original file including metadata, formatting, and musical content.

**Input:** Motif-exported JSON files.
**Output:** Unchanged JSON stored as raw data. Never modified.

### Stage 2: Intermediate Representation (02_intermediate)

The first transformation pass converts Motif JSON into a stable **event/graph format** designed for ML consumption. This is the project's intermediate representation (IR).

**What we extract (musical information only):**
- Tempo and time signature changes
- Rhythmic structure (bars, beats, note durations)
- Notes (pitch, string/fret for tablature instruments)
- Articulations and expressive markings (bends, slides, hammer-on/pull-off, tremolo, whammy bar)
- Dynamics
- Song structure and section markers
- Harmonic relationships between tracks/instruments

**What we discard:**
- Song titles, album names, artist names
- Formatting and layout directives
- Application-specific data that Motif preserves for roundtrip fidelity
- Rendering hints, display preferences, and other non-musical metadata

### Stage 3: Primary Data (03_primary)

Cleaned and normalized domain representations. Canonical source for all downstream feature extraction.

### Stage 4: Features (04_feature)

Engineered features derived from the primary data, shaped for model consumption.

### Stage 5: Model Input (05_model_input)

Tokenized sequences and tensors ready for training. See [Tokenization](#tokenization) below.

---

## Round-Trip Conversion

A core design constraint: conversions between the intermediate format and the raw Motif JSON must be **deterministic and lossless** with respect to musical content.

```
Motif JSON  ---> Intermediate IR  ---> Motif JSON
   (raw)           (02_intermediate)        (patched)
```

- **Forward (raw -> IR):** Extract musical events, discard non-musical metadata.
- **Reverse (IR -> raw):** Reconstruct a valid Motif JSON that, when parsed by Motif, produces correct patches to the original source file.
- **Invariant:** `parse(export(ir))` must yield an IR identical to the original. Musical content must survive the round trip with zero data loss.

This means the IR must preserve enough structural information to reconstruct the Motif JSON skeleton, even though we discard non-musical metadata during training. The conversion layer maintains a separation between *musical content* (used by models) and *structural scaffolding* (preserved for round-trip fidelity but invisible to the ML pipeline).

---

## Tokenization

Tokens are designed to be **musically meaningful** units rather than arbitrary subdivisions. The vocabulary includes:

### Structural Tokens
| Token Type | Description |
|---|---|
| `BAR` | Bar/measure boundary |
| `BEAT` | Beat position within a bar |
| `SECTION` | Section marker (intro, verse, chorus, bridge, solo, etc.) |
| `TEMPO` | Tempo change event |
| `TIME_SIG` | Time signature change |

### Note Event Tokens
| Token Type | Description |
|---|---|
| `NOTE_ON` | Note onset (with pitch information) |
| `NOTE_OFF` | Note release |
| `DURATION` | Note duration value |
| `REST` | Explicit rest |

### Tablature Tokens
| Token Type | Description |
|---|---|
| `STRING` | String number (for fretted instruments) |
| `FRET` | Fret position |

### Articulation & Expression Tokens
| Token Type | Description |
|---|---|
| `BEND` | Bend (with target pitch/degree) |
| `SLIDE` | Slide between notes |
| `HAMMER_ON` | Hammer-on |
| `PULL_OFF` | Pull-off |
| `TREMOLO` | Tremolo picking |
| `WHAMMY` | Whammy bar effect |
| `DYNAMIC` | Dynamic marking (pp, p, mp, mf, f, ff, etc.) |
| `ARTICULATION` | General articulation (staccato, accent, etc.) |

### Design Principles
- Each token maps to a discrete, performable musical action.
- Token sequences can be validated against musical constraints before decoding.
- The vocabulary is extensible; new token types can be added without breaking existing sequences.

---

## Model Architecture

MotifML trains two distinct transformer models, each targeting a different use case.

### Sequence Transformer (Generative Composition)

Autoregressive transformer that generates new musical sequences token by token.

- **Task:** Given a context (e.g., prior bars, style prompt, structural skeleton), generate continuation tokens.
- **Output:** New event token sequences representing composed music.
- **Training:** Next-token prediction on tokenized score data.

### Edit Transformer (Targeted Rewrite)

Encoder-decoder transformer for making targeted modifications to existing scores.

- **Task:** Given an existing score region and an edit intent, produce a modified version of that region.
- **Output:** Replacement token sequences for the targeted section.
- **Training:** Trained on paired (original, edited) score regions.
- **Use cases:** Re-harmonization, rhythmic variation, articulation changes, instrument adaptation.

### Shared Infrastructure

Both models share:
- The same tokenization vocabulary and encoding/decoding pipeline.
- Common positional encoding that respects musical time (bar-relative, beat-relative).
- Identical constraint validation on outputs.
- PyTorch implementations with config-driven architecture (layer count, head count, embedding dim, etc.).

---

## Validation & Musical Constraints

Generated outputs must be **performable by humans**. Validation is applied at multiple levels:

### Token-Level Constraints
- Every `NOTE_ON` must have a corresponding `NOTE_OFF` or `DURATION`.
- `FRET` values must be within the range of the target instrument.
- `STRING` values must be valid for the instrument's string count.
- `BEND` targets must be physically achievable on the instrument.

### Sequence-Level Constraints
- Notes within a bar must not exceed the bar's duration (as defined by time signature).
- Polyphony must not exceed what is physically playable on the instrument (e.g., max 6 simultaneous notes on a 6-string guitar).
- Articulations must be compatible with the surrounding note context (e.g., a slide requires two sequential notes).

### Score-Level Constraints
- Section structure must be coherent (no orphaned sections, valid transitions).
- Tempo and time signature changes must occur at bar boundaries.
- Track/instrument ranges must be respected.

### Validation Pipeline

Validation runs as a post-processing step on model outputs before they are converted back to the Motif format. Invalid outputs are either rejected or repaired using rule-based correction, depending on the severity of the violation.

---

## Pipeline Architecture (Kedro)

All orchestration runs through Kedro. No data processing or training logic bypasses the pipeline.

### Pipeline Composition

```
kedro run --pipeline=ingestion          # 01_raw -> 02_intermediate
kedro run --pipeline=normalization      # 02_intermediate -> 03_primary
kedro run --pipeline=feature_extraction # 03_primary -> 04_feature
kedro run --pipeline=tokenization       # 04_feature -> 05_model_input
kedro run --pipeline=training           # 05_model_input -> 06_models
kedro run --pipeline=generation         # 06_models -> 07_model_output
kedro run --pipeline=evaluation         # 07_model_output -> 08_reporting
kedro run                               # Full pipeline
```

### Node Rules
- One logical transformation per node.
- Pure functions with typed inputs and outputs.
- No direct file I/O; all data flows through the Kedro Data Catalog.
- No hardcoded paths or experimental constants.

---

## Technology Stack

| Component | Technology |
|---|---|
| Orchestration | Kedro |
| ML Framework | PyTorch |
| Environment | uv + Python 3.11 |
| Source Data | Motif (JSON export) |
| Testing | pytest |
| Linting | ruff |
| Type Checking | mypy |
| Visualization | Kedro-Viz, Jupyter |

---

## Project Layout

```
motifml/
├── src/motifml/
│   ├── pipelines/         # Kedro pipeline definitions
│   │   ├── ingestion/     # Raw JSON -> IR conversion
│   │   ├── normalization/ # IR -> normalized domain data
│   │   ├── feature_extraction/
│   │   ├── tokenization/  # Domain data -> token sequences
│   │   ├── training/      # Model training loops
│   │   ├── generation/    # Inference / generation
│   │   └── evaluation/    # Metrics and reporting
│   ├── model/             # PyTorch model architectures
│   ├── tokenizer/         # Token vocabulary and encoding
│   ├── ir/                # Intermediate representation definitions
│   ├── validation/        # Musical constraint checking
│   └── io/                # Motif JSON conversion (forward + reverse)
├── conf/
│   ├── base/              # Default configuration
│   └── local/             # Environment-specific overrides
├── data/                  # Staged data directories (gitignored)
├── tools/                 # Motif binaries (gitignored)
├── notebooks/             # Exploration and visualization
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation (this file lives here)
└── pyproject.toml
```

---

## Open Design Questions

- [ ] Exact IR schema definition (event types, graph edges, attribute encoding)
- [ ] Positional encoding strategy for musical time (absolute tick vs. bar-relative vs. beat-relative)
- [ ] Edit transformer input format (how to encode edit intent alongside source region)
- [ ] Token vocabulary sizing and special token handling (PAD, BOS, EOS, UNK)
- [ ] Training data augmentation strategies (transposition, tempo scaling, etc.)
- [ ] Constraint repair vs. rejection policy for invalid outputs
- [ ] Multi-instrument harmony encoding in token sequences (interleaved vs. parallel tracks)

---

*Last updated: 2026-03-05*
