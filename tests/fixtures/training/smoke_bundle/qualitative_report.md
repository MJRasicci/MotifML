# Baseline Evaluation Report

- Evaluation Run: `ef214e8abe56997860e7c3be52a8dd34d040c46b479f3007b3ce9d8a416cd611`
- Training Run: `59323acfe5cf05e02422f121c0601765b98b8155b047cf12ca35ff7d4eb25308`

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
- Generated Continuation: `NOTE_DURATION:96 TIME_SHIFT:32 NOTE_DURATION:96 ... (+1 more)`
