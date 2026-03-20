#!/usr/bin/env python3
"""Generate deterministic IR inspection bundles for tracked fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from motifml.ir.inspection_bundles import (
    DEFAULT_OUTPUT_ROOT,
    generate_inspection_bundles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic IR inspection bundles for tracked fixtures."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that should receive the generated inspection bundles.",
    )
    parser.add_argument(
        "--fixture-id",
        action="append",
        dest="fixture_ids",
        default=None,
        help="Optional tracked fixture id to generate. Repeat to generate multiple.",
    )
    args = parser.parse_args()

    generate_inspection_bundles(
        output_root=args.output_root,
        fixture_ids=args.fixture_ids,
    )


if __name__ == "__main__":
    main()
