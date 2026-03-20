MotifML IR Design
=================

This is the maintained ``.rst`` version of the approved IR design. It supersedes the
scratch copy in ``temp/ir_design.md`` and is written to match the repository's current
implementation status.

Purpose
-------

The MotifML IR is the canonical symbolic-score representation used inside MotifML. It is
designed to:

- preserve authored musical structure with exact rational timing
- stay format-agnostic across Motif-supported source formats
- support deterministic Kedro corpus builds
- project into sequence, graph, and hierarchical downstream views
- keep optional analytical layers separate from canonical authored truth

Implemented v1 Surface
----------------------

The repository currently implements:

- the written backbone ``Part -> Staff -> Bar -> VoiceLane -> OnsetGroup -> NoteEvent``
- score-, part-, staff-, and voice-scoped point and span controls
- sparse intrinsic edges for containment, voice succession, ties, and linked-note
  techniques
- canonical JSON serialization and a checked-in JSON schema
- structural validation and corpus summary reporting
- optional phrase overlays and typed derived-view containers
- sequence, graph, and hierarchical projection modules

The repository also includes normalization, feature-extraction, and tokenization
scaffolding built on top of the IR.

Not Yet Implemented
-------------------

The following remain intentionally outside the implemented surface:

- reverse export from IR back to Motif JSON
- training, generation, and evaluation pipelines
- playback-unrolled IR as canonical source of truth
- dense harmonic or recurrence graphs in the base document
- mandatory phrase segmentation

Canonical Document Structure
----------------------------

Each persisted IR document represents one source score and contains musical content only.
Source-relative identity stays in the manifest.

.. code-block:: text

   MotifMlIrDocument
     metadata
       ir_schema_version
       corpus_build_version
       generator_version
       source_document_hash
       time_unit = whole_note_fraction
       optional compiled_resolution_hint
     parts[]
     staves[]
     bars[]
     voice_lanes[]
     point_control_events[]
     span_control_events[]
     onset_groups[]
     note_events[]
     edges[]
     optional_overlays
       phrase_spans[]
     optional_views
       playback_instances[]
       derived_edge_sets[]

Identity and Ordering
---------------------

Determinism is enforced through document-local identifiers and canonical ordering rules.
The stable id families currently include:

- ``part:<track_id>``
- ``staff:<part_id>:<staff_index>``
- ``bar:<bar_index>``
- ``voice:<staff_id>:<bar_index>:<voice_index>``
- ``voice-chain:<part_id>:<staff_id>:<voice_index>``
- ``onset:<voice_lane_id>:<attack_index>``
- ``note:<onset_id>:<note_index>``
- ``ctrlp:<scope>:<ordinal>``
- ``ctrls:<scope>:<ordinal>``
- ``phrase:<scope>:<ordinal>``

Ordering is centralized in ``src/motifml/ir/ids.py`` and applied again during
serialization in ``src/motifml/ir/serialization.py``.

Timing Model
------------

All canonical time values use rational whole-note fractions through ``ScoreTime``.

- bar starts and durations come from the written time map
- onset groups preserve authored attack time and written rhythm shape
- note events retain both ``attack_duration`` and ``sounding_duration``
- compiled integer-time views are optional metadata, not required canonical fields

Controls, Techniques, and Edges
-------------------------------

The currently supported control kinds are:

- point controls: ``tempo_change``, ``dynamic_change``, ``fermata``
- span controls: ``hairpin``, ``ottava``

Technique payloads are split into:

- generic flags shared across instruments
- optional general payloads
- optional string-fretted payloads

Intrinsic edges stay sparse by design:

- ``contains``
- ``next_in_voice``
- ``tie_to``
- ``technique_to``

Validation and Reporting
------------------------

The IR build and validation flow treats unsupported or excluded source features as
explicit diagnostics instead of silently dropping them.

- IR build groups diagnostics into manifest summaries
- validation applies configuration-driven severities
- corpus reporting aggregates validation issues and unsupported-feature counts
- review bundles surface schema validation, IR validation, tables, and visualizations for
  human inspection

Optional Layers
---------------

Optional overlay and view support is present but not required for validity:

- phrase overlays are fully typed and serializable
- playback instances and derived edge sets are placeholder containers
- derived-view omission and explicit empties serialize canonically

Downstream Projections
----------------------

The repository includes three projection modules under ``src/motifml/ir/projections/``:

- ``sequence``: time-ordered events for sequential models
- ``graph``: typed nodes and adjacency for graph-style consumers
- ``hierarchical``: reconstructed containment trees with control attachment

The current feature-extraction scaffold selects among those projections through Kedro
parameters.
