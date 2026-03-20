"""Typed graph projection helpers for MotifML IR documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from motifml.ir.ids import (
    bar_sort_key,
    note_sort_key,
    onset_sort_key,
    part_sort_key,
    phrase_sort_key,
    point_control_sort_key,
    span_control_sort_key,
    staff_sort_key,
    voice_lane_sort_key,
)
from motifml.ir.models import (
    DerivedEdgeType,
    EdgeType,
    MotifMlIrDocument,
    NoteEvent,
)
from motifml.ir.time import ScoreTime


class GraphNodeFamily(StrEnum):
    """Canonical and optional IR node families included in graph projections."""

    PART = "part"
    STAFF = "staff"
    BAR = "bar"
    VOICE_LANE = "voice_lane"
    POINT_CONTROL = "point_control"
    SPAN_CONTROL = "span_control"
    ONSET_GROUP = "onset_group"
    NOTE_EVENT = "note_event"
    PHRASE_SPAN = "phrase_span"
    PLAYBACK_INSTANCE = "playback_instance"


class GraphEdgeFamily(StrEnum):
    """Edge provenance buckets used by the projection."""

    INTRINSIC = "intrinsic"
    DERIVED = "derived"


@dataclass(frozen=True)
class GraphProjectionParameters:
    """Configuration for graph projection edge inclusion."""

    derived_edge_types: tuple[DerivedEdgeType | str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived_edge_types",
            tuple(
                sorted(
                    {
                        _coerce_derived_edge_type(edge_type)
                        for edge_type in self.derived_edge_types
                    },
                    key=lambda edge_type: edge_type.value,
                )
            ),
        )


@dataclass(frozen=True)
class GraphNode:
    """One typed node emitted by the graph projection."""

    node_id: str
    family: GraphNodeFamily
    part_id: str | None = None
    staff_id: str | None = None
    staff_index: int | None = None
    bar_id: str | None = None
    bar_index: int | None = None
    voice_lane_id: str | None = None
    voice_lane_chain_id: str | None = None
    voice_index: int | None = None
    onset_id: str | None = None
    attack_order_in_voice: int | None = None
    note_id: str | None = None
    note_index: int | None = None
    control_id: str | None = None
    scope_ref: str | None = None
    source_ref: str | None = None
    time: ScoreTime | None = None
    end_time: ScoreTime | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _normalize_text(self.node_id, "node_id"))
        if self.label is not None:
            object.__setattr__(self, "label", _normalize_text(self.label, "label"))


GraphEdgeType: TypeAlias = EdgeType | DerivedEdgeType


@dataclass(frozen=True)
class GraphEdge:
    """One typed edge emitted by the graph projection."""

    source_id: str
    target_id: str
    edge_type: GraphEdgeType
    family: GraphEdgeFamily
    source_index: int
    target_index: int
    derived_set_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _normalize_text(self.source_id, "source_id")
        )
        object.__setattr__(
            self, "target_id", _normalize_text(self.target_id, "target_id")
        )
        if self.derived_set_name is not None:
            object.__setattr__(
                self,
                "derived_set_name",
                _normalize_text(self.derived_set_name, "derived_set_name"),
            )

        if self.source_index < 0:
            raise ValueError("source_index must be non-negative.")
        if self.target_index < 0:
            raise ValueError("target_index must be non-negative.")

        if isinstance(self.edge_type, EdgeType):
            object.__setattr__(
                self,
                "edge_type",
                EdgeType(self.edge_type),
            )
        else:
            object.__setattr__(
                self,
                "edge_type",
                _coerce_derived_edge_type(self.edge_type),
            )

        object.__setattr__(self, "family", GraphEdgeFamily(self.family))

    def sort_key(self) -> tuple[int, str, str, str, str]:
        """Return a stable canonical order for graph edges."""
        return (
            self.family.value,
            self.edge_type.value,
            "" if self.derived_set_name is None else self.derived_set_name.casefold(),
            self.source_id,
            self.target_id,
        )


@dataclass(frozen=True)
class GraphAdjacency:
    """Adjacency structures ready for graph neural network consumption."""

    node_ids: tuple[str, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_types: tuple[str, ...]
    outgoing_by_node: tuple[tuple[int, ...], ...]
    incoming_by_node: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        node_ids = tuple(
            _normalize_text(node_id, "node_ids entry") for node_id in self.node_ids
        )
        object.__setattr__(self, "node_ids", node_ids)
        if len(self.edge_index) != _EDGE_INDEX_COMPONENTS:
            raise ValueError(
                "edge_index must contain source and target index sequences."
            )

        source_indices = tuple(int(index) for index in self.edge_index[0])
        target_indices = tuple(int(index) for index in self.edge_index[1])
        if len(source_indices) != len(target_indices) or len(source_indices) != len(
            self.edge_types
        ):
            raise ValueError("edge_index and edge_types must have matching lengths.")

        object.__setattr__(self, "edge_index", (source_indices, target_indices))
        object.__setattr__(self, "edge_types", tuple(self.edge_types))
        object.__setattr__(
            self,
            "outgoing_by_node",
            tuple(
                tuple(int(index) for index in neighbors)
                for neighbors in self.outgoing_by_node
            ),
        )
        object.__setattr__(
            self,
            "incoming_by_node",
            tuple(
                tuple(int(index) for index in neighbors)
                for neighbors in self.incoming_by_node
            ),
        )
        if len(self.outgoing_by_node) != len(node_ids) or len(
            self.incoming_by_node
        ) != len(node_ids):
            raise ValueError("Adjacency lists must contain one entry per node.")


@dataclass(frozen=True)
class GraphProjection:
    """Typed graph representation of one MotifML IR document."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    adjacency: GraphAdjacency
    node_index_by_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        normalized_nodes = tuple(self.nodes)
        normalized_edges = tuple(self.edges)
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", normalized_edges)
        node_index_by_id = {
            node.node_id: index for index, node in enumerate(normalized_nodes)
        }
        if len(node_index_by_id) != len(normalized_nodes):
            raise ValueError("Graph node ids must be unique.")

        object.__setattr__(self, "node_index_by_id", node_index_by_id)
        if self.adjacency.node_ids != tuple(node.node_id for node in normalized_nodes):
            raise ValueError("Adjacency node ordering must match the projected nodes.")

        if self.adjacency.edge_index[0] != tuple(
            edge.source_index for edge in normalized_edges
        ):
            raise ValueError("Adjacency source indices must match the projected edges.")

        if self.adjacency.edge_index[1] != tuple(
            edge.target_index for edge in normalized_edges
        ):
            raise ValueError("Adjacency target indices must match the projected edges.")

    @property
    def node_count(self) -> int:
        """Return the number of projected nodes."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of projected edges."""
        return len(self.edges)


def project_graph(
    document: MotifMlIrDocument,
    parameters: GraphProjectionParameters | None = None,
) -> GraphProjection:
    """Project a canonical IR document into a typed graph."""
    projection_parameters = parameters or GraphProjectionParameters()
    nodes = _collect_nodes(document)
    node_index_by_id = {node.node_id: index for index, node in enumerate(nodes)}
    edges = _collect_edges(document, node_index_by_id, projection_parameters)
    adjacency = _build_adjacency(nodes, edges)
    return GraphProjection(nodes=nodes, edges=edges, adjacency=adjacency)


def _collect_nodes(document: MotifMlIrDocument) -> tuple[GraphNode, ...]:
    bar_index_by_id = {bar.bar_id: bar.bar_index for bar in document.bars}
    staff_index_by_id = {staff.staff_id: staff.staff_index for staff in document.staves}
    voice_lane_by_id = {
        voice_lane.voice_lane_id: voice_lane for voice_lane in document.voice_lanes
    }

    nodes: list[GraphNode] = []

    for part in sorted(document.parts, key=lambda item: part_sort_key(item.part_id)):
        nodes.append(
            GraphNode(
                node_id=part.part_id,
                family=GraphNodeFamily.PART,
                part_id=part.part_id,
                label=part.part_id,
            )
        )

    for staff in sorted(
        document.staves,
        key=lambda item: staff_sort_key(item.part_id, item.staff_index, item.staff_id),
    ):
        nodes.append(
            GraphNode(
                node_id=staff.staff_id,
                family=GraphNodeFamily.STAFF,
                part_id=staff.part_id,
                staff_id=staff.staff_id,
                staff_index=staff.staff_index,
                label=f"staff:{staff.staff_index}",
            )
        )

    for bar in sorted(
        document.bars, key=lambda item: bar_sort_key(item.bar_index, item.bar_id)
    ):
        nodes.append(
            GraphNode(
                node_id=bar.bar_id,
                family=GraphNodeFamily.BAR,
                bar_id=bar.bar_id,
                bar_index=bar.bar_index,
                time=bar.start,
                end_time=bar.start + bar.duration,
                label=f"bar:{bar.bar_index}",
            )
        )

    for voice_lane in sorted(
        document.voice_lanes,
        key=lambda item: voice_lane_sort_key(
            bar_index_by_id[item.bar_id],
            item.staff_id,
            item.voice_index,
            item.voice_lane_id,
        ),
    ):
        nodes.append(
            GraphNode(
                node_id=voice_lane.voice_lane_id,
                family=GraphNodeFamily.VOICE_LANE,
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                staff_index=staff_index_by_id[voice_lane.staff_id],
                bar_id=voice_lane.bar_id,
                bar_index=bar_index_by_id[voice_lane.bar_id],
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                label=f"voice:{voice_lane.voice_index}",
            )
        )

    for control in sorted(
        document.point_control_events,
        key=lambda item: point_control_sort_key(
            item.scope.value, item.target_ref, item.time, item.control_id
        ),
    ):
        nodes.append(
            GraphNode(
                node_id=control.control_id,
                family=GraphNodeFamily.POINT_CONTROL,
                control_id=control.control_id,
                scope_ref=control.target_ref,
                time=control.time,
                label=control.kind.value,
            )
        )

    for control in sorted(
        document.span_control_events,
        key=lambda item: span_control_sort_key(
            item.scope.value,
            item.target_ref,
            item.start_time,
            item.end_time,
            item.control_id,
        ),
    ):
        nodes.append(
            GraphNode(
                node_id=control.control_id,
                family=GraphNodeFamily.SPAN_CONTROL,
                control_id=control.control_id,
                scope_ref=control.target_ref,
                time=control.start_time,
                end_time=control.end_time,
                label=control.kind.value,
            )
        )

    sorted_onsets = sorted(
        document.onset_groups,
        key=lambda item: onset_sort_key(
            item.voice_lane_id, item.time, item.attack_order_in_voice, item.onset_id
        ),
    )
    notes_by_onset: dict[str, list[NoteEvent]] = {}
    for note in document.note_events:
        notes_by_onset.setdefault(note.onset_id, []).append(note)

    for onset in sorted_onsets:
        nodes.append(
            GraphNode(
                node_id=onset.onset_id,
                family=GraphNodeFamily.ONSET_GROUP,
                part_id=voice_lane_by_id[onset.voice_lane_id].part_id,
                staff_id=voice_lane_by_id[onset.voice_lane_id].staff_id,
                staff_index=staff_index_by_id[
                    voice_lane_by_id[onset.voice_lane_id].staff_id
                ],
                bar_id=onset.bar_id,
                bar_index=bar_index_by_id[onset.bar_id],
                voice_lane_id=onset.voice_lane_id,
                voice_lane_chain_id=voice_lane_by_id[
                    onset.voice_lane_id
                ].voice_lane_chain_id,
                time=onset.time,
                attack_order_in_voice=onset.attack_order_in_voice,
                label="rest" if onset.is_rest else "attack",
            )
        )

        ordered_notes = sorted(
            notes_by_onset.get(onset.onset_id, ()),
            key=lambda item: note_sort_key(
                item.string_number, item.pitch, item.note_id
            ),
        )
        for note_index, note in enumerate(ordered_notes):
            nodes.append(
                GraphNode(
                    node_id=note.note_id,
                    family=GraphNodeFamily.NOTE_EVENT,
                    part_id=note.part_id,
                    staff_id=note.staff_id,
                    staff_index=staff_index_by_id[note.staff_id],
                    bar_id=onset.bar_id,
                    bar_index=bar_index_by_id[onset.bar_id],
                    voice_lane_id=onset.voice_lane_id,
                    voice_lane_chain_id=voice_lane_by_id[
                        onset.voice_lane_id
                    ].voice_lane_chain_id,
                    onset_id=note.onset_id,
                    note_id=note.note_id,
                    note_index=note_index,
                    time=note.time,
                    label=_render_note_label(note),
                )
            )

    for phrase_span in sorted(
        document.optional_overlays.phrase_spans,
        key=lambda item: phrase_sort_key(
            item.scope_ref, item.start_time, item.end_time, item.phrase_id
        ),
    ):
        nodes.append(
            GraphNode(
                node_id=phrase_span.phrase_id,
                family=GraphNodeFamily.PHRASE_SPAN,
                scope_ref=phrase_span.scope_ref,
                voice_lane_chain_id=phrase_span.voice_lane_chain_id,
                time=phrase_span.start_time,
                end_time=phrase_span.end_time,
                label=phrase_span.phrase_kind.value,
            )
        )

    for playback_instance in document.optional_views.playback_instances:
        nodes.append(
            GraphNode(
                node_id=playback_instance.instance_id,
                family=GraphNodeFamily.PLAYBACK_INSTANCE,
                voice_lane_chain_id=playback_instance.voice_lane_chain_id,
                source_ref=playback_instance.source_ref,
                time=playback_instance.start_time,
                end_time=playback_instance.end_time,
                label=playback_instance.source_ref,
            )
        )

    return tuple(sorted(nodes, key=_node_sort_key))


def _collect_edges(
    document: MotifMlIrDocument,
    node_index_by_id: dict[str, int],
    parameters: GraphProjectionParameters,
) -> tuple[GraphEdge, ...]:
    selected_derived_edge_types = set(parameters.derived_edge_types)
    edges: list[GraphEdge] = []

    for edge in sorted(
        document.edges,
        key=lambda item: (item.source_id, item.edge_type.value, item.target_id),
    ):
        edges.append(
            _build_graph_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.edge_type,
                family=GraphEdgeFamily.INTRINSIC,
                node_index_by_id=node_index_by_id,
            )
        )

    for derived_set in document.optional_views.derived_edge_sets:
        for edge in derived_set.edges:
            if edge.edge_type not in selected_derived_edge_types:
                continue

            edges.append(
                _build_graph_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    family=GraphEdgeFamily.DERIVED,
                    node_index_by_id=node_index_by_id,
                    derived_set_name=derived_set.name,
                )
            )

    return tuple(sorted(edges, key=lambda item: item.sort_key()))


def _build_graph_edge(  # noqa: PLR0913
    *,
    source_id: str,
    target_id: str,
    edge_type: EdgeType | DerivedEdgeType,
    family: GraphEdgeFamily,
    node_index_by_id: dict[str, int],
    derived_set_name: str | None = None,
) -> GraphEdge:
    try:
        source_index = node_index_by_id[source_id]
    except KeyError as exc:
        raise ValueError(
            f"Graph edge source '{source_id}' does not reference a projected node."
        ) from exc

    try:
        target_index = node_index_by_id[target_id]
    except KeyError as exc:
        raise ValueError(
            f"Graph edge target '{target_id}' does not reference a projected node."
        ) from exc

    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        family=family,
        source_index=source_index,
        target_index=target_index,
        derived_set_name=derived_set_name,
    )


def _build_adjacency(
    nodes: tuple[GraphNode, ...], edges: tuple[GraphEdge, ...]
) -> GraphAdjacency:
    outgoing: list[list[int]] = [[] for _ in nodes]
    incoming: list[list[int]] = [[] for _ in nodes]
    source_indices: list[int] = []
    target_indices: list[int] = []
    edge_types: list[str] = []

    for edge in edges:
        source_indices.append(edge.source_index)
        target_indices.append(edge.target_index)
        edge_types.append(edge.edge_type.value)
        outgoing[edge.source_index].append(edge.target_index)
        incoming[edge.target_index].append(edge.source_index)

    return GraphAdjacency(
        node_ids=tuple(node.node_id for node in nodes),
        edge_index=(tuple(source_indices), tuple(target_indices)),
        edge_types=tuple(edge_types),
        outgoing_by_node=tuple(tuple(indices) for indices in outgoing),
        incoming_by_node=tuple(tuple(indices) for indices in incoming),
    )


def _node_sort_key(node: GraphNode) -> tuple[object, ...]:  # noqa: PLR0911
    family_order = _NODE_FAMILY_ORDER[node.family]
    if node.family is GraphNodeFamily.PART:
        return (family_order, node.node_id)
    if node.family is GraphNodeFamily.STAFF:
        return (family_order, node.part_id, node.staff_index, node.node_id)
    if node.family is GraphNodeFamily.BAR:
        return (family_order, node.bar_index, node.node_id)
    if node.family is GraphNodeFamily.VOICE_LANE:
        return (
            family_order,
            node.bar_index,
            node.staff_id,
            node.voice_index,
            node.node_id,
        )
    if node.family is GraphNodeFamily.POINT_CONTROL:
        return (family_order, node.scope_ref, node.time, node.node_id)
    if node.family is GraphNodeFamily.SPAN_CONTROL:
        return (
            family_order,
            node.scope_ref,
            node.time,
            node.end_time,
            node.node_id,
        )
    if node.family is GraphNodeFamily.ONSET_GROUP:
        return (
            family_order,
            node.bar_index,
            node.voice_lane_id,
            node.time,
            node.attack_order_in_voice,
            node.node_id,
        )
    if node.family is GraphNodeFamily.NOTE_EVENT:
        return (
            family_order,
            node.onset_id,
            node.note_index,
            node.node_id,
        )
    if node.family is GraphNodeFamily.PHRASE_SPAN:
        return (
            family_order,
            node.scope_ref,
            node.time,
            node.end_time,
            node.node_id,
        )
    return (family_order, node.time, node.end_time, node.node_id)


def _render_note_label(note: NoteEvent) -> str:
    if note.pitch is None:
        return note.note_id

    accidental = "" if note.pitch.accidental is None else note.pitch.accidental
    return f"{note.pitch.step.value}{accidental}{note.pitch.octave}"


def _coerce_derived_edge_type(value: DerivedEdgeType | str) -> DerivedEdgeType:
    if isinstance(value, DerivedEdgeType):
        return value

    return DerivedEdgeType(value)


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


_NODE_FAMILY_ORDER = {
    GraphNodeFamily.PART: 0,
    GraphNodeFamily.STAFF: 1,
    GraphNodeFamily.BAR: 2,
    GraphNodeFamily.VOICE_LANE: 3,
    GraphNodeFamily.POINT_CONTROL: 4,
    GraphNodeFamily.SPAN_CONTROL: 5,
    GraphNodeFamily.ONSET_GROUP: 6,
    GraphNodeFamily.NOTE_EVENT: 7,
    GraphNodeFamily.PHRASE_SPAN: 8,
    GraphNodeFamily.PLAYBACK_INSTANCE: 9,
}

_EDGE_INDEX_COMPONENTS = 2
