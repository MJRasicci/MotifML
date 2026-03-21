# Baseline Evaluation Report

- Evaluation Run: `241ec6cf17f06c80e769452756053609adb31123d85275624438c6df6202bbc8`
- Training Run: `eb8ed8dd6235aa5906335694f9bde070bebab3ab0c1ff08a874b0ad07827bc5f`

## Validation

### Quantitative Metrics

- Cross-Entropy Loss: 3.133404
- Perplexity: 22.951976
- Accuracy: 0.041096
- Top-3 Accuracy: 0.191781
- Evaluation `<unk>` Rate: 0.346667 (26/75, max 0.500000)

### Structural Checks

- Valid Transition Rate: 0.666667
- Boundary-Order Pass Rate: 1.000000
- Generated `<unk>` Rate: 0.000000 (0/4, max 0.500000)
- Pitch Out-of-Range Fraction: 0.000000
- Duration Distribution TV Distance: 1.000000

### Samples

#### Sample 1: `ensemble_polyphony_controls.json`

- Document ID: `ensemble_polyphony_controls.json`
- Prompt: `<bos> STRUCTURE:PART STRUCTURE:PART ... (+1 more)`
- Reference Continuation: `STRUCTURE:STAFF STRUCTURE:STAFF STRUCTURE:BAR ... (+1 more)`
- Generated Continuation: `NOTE_DURATION:2882880 TIME_SHIFT:960960 NOTE_DURATION:2882880 ... (+1 more)`
