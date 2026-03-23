V1 Continuation Task Contract
=============================

This page freezes the first recoverable generation task for MotifML. The repository does
not yet implement this path as a first-class Kedro pipeline, but downstream dataset
extraction, model-input design, decoding, evaluation, and reporting work should treat
this document as the authoritative V1 target rather than inferring task rules from the
baseline document-row training stack.

The canonical implementation target for this contract is a first-class Kedro-owned
training and evaluation path, not a notebook-only continuation helper.

Contract Identity
-----------------

- contract name: ``motifml.v1_continuation_task``
- contract version: ``1.0.0``
- canonical task: ``single-track, single-voice, next-bar continuation``

Any downstream persisted surface that depends on the legal task state space, prompt
construction, scaffold representation, or bar-completion rules should derive a task-level
version key from this contract rather than only from baseline tokenization settings.

Canonical Unit of Generation
----------------------------

One V1 example represents exactly one target bar from one normalized IR source document.
The example is defined over:

- one source score that contributes exactly one musical track / part to the example
- one deterministic voice-lane chain inside that track
- four complete prompt bars that immediately precede the target bar
- one scaffolded target bar whose structure is known up front
- one predicted continuation payload that fills that scaffold

The model is not asked to continue an entire document, invent phrase structure, or
jointly coordinate multiple tracks. V1 success means filling one known next-bar slot
reliably.

Eligibility Rules
-----------------

The canonical V1 dataset and benchmark only admit examples that satisfy all of these
conditions:

- the source example is drawn from a single-track score surface
- the source example contains one active voice-lane chain for the selected material
- at least four complete prompt bars exist immediately before the target bar
- the target bar is a complete written bar with stable bar duration metadata
- the target bar can be reduced into an ordered onset template without ambiguity
- the prompt bars and target bar share one stable track identity and one stable
  voice-lane identity

Scores that require multi-track coordination, multiple simultaneous voice lanes, or
special-case structural repair are outside the canonical V1 extraction domain.

Prompt Context
--------------

The prompt is the four immediately preceding complete bars from the same track and
voice-lane chain as the target bar. The prompt window is fixed:

- prompt length: exactly four complete bars
- prompt ordering: oldest to newest, preserving bar order
- prompt source: copied from normalized IR without target-bar leakage
- prompt eligibility: examples with fewer than four prior complete bars are excluded from
  the canonical V1 dataset and benchmark

Training and inference must use the same prompt selection rule. V1 does not support
variable-length prompt windows as the canonical path.

Target Scaffold
---------------

The target bar is scaffold-conditioned rather than free-form. The scaffold exposes the
structural shape of the next bar while withholding the musical content that the model must
predict.

The scaffold copies these fields from the target bar:

- target bar identity and relative ordering metadata
- written bar duration and meter needed to interpret bar-local offsets
- ordered onset slots for the target bar
- the bar-local offset of each onset slot
- the required note-slot count inside each onset slot
- deterministic slot indices for onset order and note order within each onset

The scaffold does not copy target musical content that the model is supposed to predict.
In particular, it must not reveal target pitches or target durations.

Predicted vs Copied Fields
--------------------------

V1 predicts only the continuation content required to fill the scaffold. The intended
task split is:

Copied / conditioned
   prompt bars, target-bar timing geometry, onset-slot offsets, note-slot counts, and
   any structural delimiters needed to identify where the target fill begins and ends.

Predicted
   one pitch payload and one duration payload for every note slot in scaffold order.

Boundary events with fixed semantics
   close-bar and example-EOS events. The concrete serializer may emit them explicitly or
   inject them deterministically, but their placement and legality are frozen by this
   contract rather than left to ad hoc notebook behavior.

Derived outside musical-content prediction
   neutral/default export-only fields and any metadata needed only for tracing,
   reporting, or reconstruction.

V1 does not ask the model to predict tempo, key changes, dynamics, articulation,
string/fingering metadata, multi-bar form, or structural-control events.

Rhythm Policy
-------------

V1 uses a partially scaffold-constrained rhythm contract.

- onset placement is scaffold-constrained
- simultaneous note-slot cardinality is scaffold-constrained
- note durations are predicted
- the model is not responsible for inventing new onset positions or new structural slots

This means V1 learns how to fill a known next-bar template, not how to invent the next
bar's topology from scratch.

Legal Output Space
------------------

The legal V1 output space is defined as an event grammar, not by one specific token-string
encoding. Later representation work may choose the concrete token names, but it must not
change the task grammar frozen here.

One legal V1 example has this structure:

.. code-block:: text

   prompt_bar_1
   prompt_bar_2
   prompt_bar_3
   prompt_bar_4
   target_bar_start
   target_onset_slot_0
     target_note_slot_0_0 -> predict pitch, predict duration
     ...
   target_onset_slot_1
     ...
   ...
   close_target_bar
   example_eos

The legal-event rules are:

- exactly one target bar is generated per example
- target onset slots are consumed strictly in scaffold order
- target note slots are consumed strictly in scaffold order within each onset
- every target note slot receives exactly one pitch prediction and exactly one positive
  duration prediction
- no extra onset slots, note slots, tracks, voices, or control events may be introduced
- ``close_target_bar`` is illegal until every scaffold slot has been filled and the bar
  is complete under the rules below
- ``example_eos`` is illegal before ``close_target_bar`` and must terminate the example

Whether ``close_target_bar`` and ``example_eos`` are explicit serialized events or
deterministic boundary insertions is a representation-layer choice. That choice must stay
identical between training and inference.

Bar Completion Semantics
------------------------

A target bar counts as complete only when all of the following are true:

- every scaffolded onset slot has been visited
- every scaffolded note slot has exactly one predicted pitch and one predicted duration
- every predicted duration is positive
- no predicted note end extends beyond the target bar duration
- the farthest predicted note end lands exactly on the target bar boundary
- the example emits one explicit close-bar event after the target bar is complete

Underfilled bars, overfilled bars, early EOS, or outputs that leave unresolved scaffold
slots are invalid V1 generations.

Training and Inference Contract
-------------------------------

Training-time and inference-time examples must use the same representation family and the
same task rules.

Training must:

- build the model input from ``prompt context + target scaffold + reference target fill``
- apply loss only to the predicted target-fill region, not to copied prompt or copied
  scaffold content
- use the same slot ordering, boundary markers, and completion semantics that inference
  will use

Inference and evaluation must:

- build the model input from ``prompt context + target scaffold`` with no hidden access
  to the withheld target fill
- use the same legality rules and close-bar semantics as training
- treat the scaffold as required input rather than something the model infers implicitly

Benchmark evaluation may derive the scaffold from the held-out reference bar, but only as
conditioning context. The scored output remains the predicted fill for that bar.

Required Reporting Implications
-------------------------------

Any V1 dataset, decoding, or evaluation surface derived from this contract should report:

- how many candidate examples satisfied the eligibility rules
- how many examples were rejected and why
- whether prompt/scaffold construction matched the frozen contract
- whether generated outputs reached legal close-bar completion

These reports are part of the task contract because silent eligibility drift or silent
decoder repair would make V1 results hard to trust.

Non-Goals
---------

The following capabilities are explicitly out of scope for V1:

- multi-track generation
- multiple simultaneous voice lanes or voice-lane handoff handling
- free-form invention of onset topology or bar structure
- generation of more than one target bar per example
- phrase planning, section planning, or global-form invention
- prediction of expressive controls, articulations, or format-specific metadata
- unconstrained recovery from arbitrary ensemble scores
- any training or inference path that relies on task rules unavailable at evaluation time

The purpose of V1 is to prove that MotifML can reliably fill one known next-bar template
before the project attempts broader continuation or richer generation behavior.
