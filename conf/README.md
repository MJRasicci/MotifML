# Configuration

This directory holds Kedro configuration for MotifML.

## Base Configuration

Shared project configuration lives in `conf/base/`.

- `catalog.yml` wires the staged corpus, IR, reporting, feature, and model-input
  datasets.
- `parameters.yml` defines deterministic IR build metadata, validation severities,
  projection settings, tokenization parameters, and the frozen training-phase families
  for split planning, sequence schema, vocabulary construction, model input, model
  architecture, training, evaluation, and seed control.
- `logging.yml` contains the shared Kedro logging setup.

Keep variable behavior here instead of hardcoding it in pipeline code.

## Local Configuration

Use `conf/local/` for machine-specific or sensitive overrides such as credentials,
private paths, or developer-only settings.

Do not commit local configuration.

## Overrides

For one-off experiments, prefer Kedro parameter overrides instead of editing the shared
base files:

```bash
uv run kedro run --params feature_extraction.projection_type=graph
```

Training-oriented overrides use the same surface:

```bash
uv run kedro run --params training.learning_rate=0.0001
```

See the Kedro configuration docs for loader and merge behavior:
https://docs.kedro.org/en/stable/configure/configuration_basics/
