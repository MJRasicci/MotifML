MotifML IR Contract
===================

This document records the current canonical IR surface implemented in
``src/motifml/ir/`` and by the ``ir_build`` pipeline.

Design Goals
------------

The MotifML IR is the repository's canonical symbolic-score representation. It is
designed to:

- preserve authored written structure with exact rational timing
- remain format-agnostic once source data has entered Motif JSON
- support deterministic Kedro corpus builds and byte-stable persistence
- separate canonical musical content from optional analytical overlays and views
- provide stable projection surfaces for downstream sequential, graph, and hierarchical
  consumers

Implemented v1 Surface
----------------------

The current repository implements:

- the written structural backbone
  ``Part -> Staff -> Bar -> VoiceLane -> OnsetGroup -> NoteEvent``
- score-, part-, staff-, and voice-scoped point and span controls
- sparse intrinsic edges for containment, voice succession, ties, and linked-note
  techniques
- canonical JSON serialization and a checked-in JSON schema
- manifest-level conversion diagnostics and rule-based structural validation
- optional phrase overlays and optional derived-view containers
- sequence, graph, and hierarchical projection modules built directly from canonical IR

Canonical Document Shape
------------------------

Each persisted IR document represents one score and contains musical content only.
Source-relative identity, build timestamps, and grouped conversion diagnostics live in
the IR manifest rather than in the document body.

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

The metadata payload is intentionally narrow. Free-form source metadata such as title,
artist, album, comments, or track names is forbidden in persisted IR documents and is
checked explicitly by the validation layer.

Identity and Canonical Ordering
-------------------------------

Determinism depends on stable, document-local identifiers derived from source position.
The implemented identifier families include:

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

Canonical ordering is centralized in ``src/motifml/ir/ids.py`` and re-applied during
serialization in ``src/motifml/ir/serialization.py``. Persisted JSON is therefore
byte-stable for identical logical content.

Timing and Rhythm Model
-----------------------

All canonical time values use ``ScoreTime`` rational whole-note fractions.

The implemented timing model preserves:

- bar starts and bar durations from the written time map
- onset attack time and notated duration
- per-note attack duration and sounding duration
- optional onset rhythm-shape metadata, including dots and tuplet ratios
- optional compiled integer-time hints in document metadata when needed by downstream
  consumers

This keeps the canonical IR faithful to authored notation while still allowing derived
integer-resolution views to be layered on top later.

Structural Entities
-------------------

The core structural dataclasses model different levels of written score organization:

``Part``
   Track-level musical context, including instrument family, kind, role, transposition,
   and owned staff identifiers.

``Staff``
   Staff-local context such as tuning pitches or capo information when present.

``Bar``
   Score-wide written bar geometry, meter, key-signature context, triplet feel, and
   optional pickup-bar context.

``VoiceLane``
   Bar-scoped authored voice lanes, plus a deterministic voice-lane chain identifier so
   voice dropouts and reentries can be tracked across bars.

``OnsetGroup``
   Authored attack or rest slots within a voice lane, including local techniques,
   dynamics, grace-note classification, and rhythm shape.

``NoteEvent``
   Notes attached to onsets, including pitch, durations, velocity, optional string
   number display, and note-local technique payloads.

Controls and Techniques
-----------------------

The current point-control kinds are:

- ``tempo_change``
- ``dynamic_change``
- ``fermata``

The current span-control kinds are:

- ``hairpin``
- ``ottava``

Control scope is explicit and typed as ``score``, ``part``, ``staff``, or ``voice``.
Control payloads are modeled as dataclasses rather than free-form dictionaries.

Technique payloads are split into three layers:

- generic flags shared across instruments
- optional general technique payloads
- optional string-fretted technique payloads

This preserves type safety without forcing every future instrument family into the base
document prematurely.

Relations and Optional Layers
-----------------------------

Intrinsic edges stored in the canonical document stay intentionally sparse:

- ``contains``
- ``next_in_voice``
- ``tie_to``
- ``technique_to``

The base document may also carry optional overlay and view containers:

- phrase overlays in ``optional_overlays.phrase_spans``
- playback-instance placeholders in ``optional_views.playback_instances``
- named derived-edge sets in ``optional_views.derived_edge_sets``

These layers are optional by design. Their presence is explicit, their omission is also
canonical, and they do not redefine the base written-score contract.

Manifest and Validation Contract
--------------------------------

The IR manifest complements persisted documents with source-relative and build-level
metadata:

- source path and source hash
- IR artifact path
- explicit build timestamp from Kedro parameters
- node and edge counts
- grouped conversion diagnostics
- quick counts of unsupported or intentionally excluded source features

Structural validation is rule-based and currently checks:

- onset ownership
- note ownership
- note time alignment
- voice-lane onset timing
- attack-order contiguity
- positive sounding duration
- linear tie chains
- voice-lane chain stability
- canonical note ordering
- edge endpoint integrity
- absence of forbidden source metadata
- phrase-span validity
- fretted-string collisions

Per-rule severities are configuration-driven through ``conf/base/parameters.yml``.

Downstream Projections
----------------------

MotifML currently exposes three typed projection families under
``src/motifml/ir/projections/``:

``sequence``
   Emits a time-ordered event stream. The current implementation supports
   ``notes_only``, ``notes_and_controls``, and
   ``notes_and_controls_and_structure_markers`` modes.

``graph``
   Emits typed nodes, typed edges, and adjacency structures suitable for graph-based
   consumers. Intrinsic edges are always available, and optional derived-edge families
   can be requested through projection parameters.

``hierarchical``
   Reconstructs containment trees over parts, staves, bars, voice lanes, onsets, and
   notes, with scoped controls attached at the appropriate structural level.

Not Yet Implemented
-------------------

The following remain outside the implemented canonical IR surface:

- reverse export from IR back to Motif JSON
- generation pipelines
- a playback-unrolled representation as the canonical source of truth
- dense harmonic, recurrence, or analytical graphs in the base document
- mandatory phrase segmentation as a validity requirement

Those capabilities may be added later, but they are not part of the repository's
current documented contract.
