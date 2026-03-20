"""Shared constants for the tracked IR fixture catalog."""

from __future__ import annotations

PENDING_GOLDEN_REVIEW_STATUS = "pending_review"
APPROVED_GOLDEN_REVIEW_STATUS = "approved"

VALID_GOLDEN_REVIEW_STATUSES = (
    PENDING_GOLDEN_REVIEW_STATUS,
    APPROVED_GOLDEN_REVIEW_STATUS,
)


def normalize_golden_review_status(review_status: str) -> str:
    """Validate and normalize a fixture review status."""
    if review_status not in VALID_GOLDEN_REVIEW_STATUSES:
        raise ValueError(
            f"Unsupported golden review status '{review_status}'. "
            f"Expected one of {VALID_GOLDEN_REVIEW_STATUSES}."
        )

    return review_status
