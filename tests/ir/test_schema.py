"""Contract tests for the persisted IR JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from motifml.ir.serialization import deserialize_document, serialize_document

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "motifml"
    / "ir"
    / "schema"
    / "motifml-ir-document.schema.json"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "ir"
    / "representative_document.ir.json"
)


def test_ir_schema_is_a_valid_draft_2020_12_schema():
    schema = _load_json(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)


def test_serialized_ir_documents_validate_against_the_checked_in_schema():
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    fixture_payload = _load_json(FIXTURE_PATH)
    canonical_document = deserialize_document(FIXTURE_PATH.read_text(encoding="utf-8"))
    serialized_payload = json.loads(serialize_document(canonical_document))

    assert list(validator.iter_errors(fixture_payload)) == []
    assert list(validator.iter_errors(serialized_payload)) == []


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
