## Summary

- Describe the IR or docs change.

## Verification

- List the commands you ran.

## IR Review Checklist

- [ ] Determinism remains intact for the affected IR output surfaces.
- [ ] Forbidden metadata is still excluded from persisted IR documents.
- [ ] Fixture, golden, and review-bundle artifacts were regenerated with the tracked
      tools when needed.
- [ ] Validation coverage was added or updated for new mapping paths, payloads, or
      unsupported-feature cases.
- [ ] Unsupported or excluded source features remain visible in manifest, summary, or
      review-bundle outputs.
- [ ] I either reviewed at least one changed golden artifact or review bundle with a
      human reviewer, or this PR does not change persisted IR shape.

## Human-Reviewed Artifact

- If persisted IR shape changed, name the reviewed fixture or bundle and the resulting
  review status in `tests/fixtures/ir_fixture_catalog.json`.
- If persisted IR shape did not change, say `no persisted IR shape change`.
