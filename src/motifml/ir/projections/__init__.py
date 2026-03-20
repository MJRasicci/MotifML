"""Typed downstream projections built on canonical MotifML IR documents."""

from motifml.ir.projections.graph import (
    GraphProjection,
    GraphProjectionParameters,
    project_graph,
)
from motifml.ir.projections.hierarchical import (
    HierarchicalProjection,
    project_hierarchical,
)
from motifml.ir.projections.sequence import (
    SequenceProjection,
    SequenceProjectionConfig,
    SequenceProjectionMode,
    project_sequence,
)

__all__ = [
    "GraphProjection",
    "GraphProjectionParameters",
    "HierarchicalProjection",
    "SequenceProjection",
    "SequenceProjectionConfig",
    "SequenceProjectionMode",
    "project_graph",
    "project_hierarchical",
    "project_sequence",
]
