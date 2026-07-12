# Eight-Phase Traffic-Light Actions Design

## Context

The current controller uses four canonical phases. Each phase serves a pair of compatible approach groups:

- Phase 0: slots 0/2 through and right.
- Phase 1: slots 0/2 left, with slots 1/3 right permissive.
- Phase 2: slots 1/3 through and right.
- Phase 3: slots 1/3 left, with slots 0/2 right permissive.

This works for balanced traffic, but it gives the policy no direct way to fully clear one heavily backlogged approach. The new design adds protected single-approach clearance phases while preserving the existing phase semantics.

## Goals

- Expand the action space from 4 phases to 8 phases.
- Keep phases 0-3 unchanged so current logs and behavior remain interpretable.
- Add one protected "open this slot" phase per canonical approach slot.
- Keep one fixed action space for both 3-way and 4-way junctions.
- Avoid action masks for this change.

## Phase Semantics

The fixed 8-action phase table is:

| Phase | Meaning |
| --- | --- |
| 0 | Existing slots 0/2 through + right phase |
| 1 | Existing slots 0/2 left phase with slots 1/3 permissive right |
| 2 | Existing slots 1/3 through + right phase |
| 3 | Existing slots 1/3 left phase with slots 0/2 permissive right |
| 4 | Slot 0 protected left + straight + right |
| 5 | Slot 1 protected left + straight + right |
| 6 | Slot 2 protected left + straight + right |
| 7 | Slot 3 protected left + straight + right on 4-way junctions; all-red pause on 3-way junctions |

For 3-way junctions, slot 3 is absent. Phase 7 is intentionally retained as an all-red pause rather than being masked or remapped to another traffic movement. This keeps the output space fixed and makes the exceptional behavior explicit.

## Architecture Changes

The canonical phase definitions are currently duplicated in `scripts/build_network.py` and `src/environment/junction_info.py`. Both locations need the same 8-phase mapping:

- Existing phase assignments remain unchanged.
- For each movement from slot `N`, add protected green `G` to phase `4 + N`.
- Phase strings are generated with eight green states instead of four.
- Yellow states and all-red states continue to be derived mechanically from green states.

`JunctionInfo.phase_states`, `yellow_states`, and `phase_served_lanes` become length 8. `conn_to_phase` should be replaced or extended with a connection-to-phases structure so expert scoring can credit every phase that protects a movement, not just the first protected phase. This matters because a movement may be served by both an existing paired phase and a new single-slot phase.

## Observation And Model Dimensions

The observation vector changes from 41 to 45 dimensions:

- 36 movement features remain unchanged.
- Current phase one-hot expands from 4 dimensions to 8 dimensions.
- Elapsed phase time remains one scalar.

The policy output head changes from 4 classes to 8 classes. Existing checkpoints with 4-action heads are not shape-compatible with the new model without migration logic, so retraining is expected.

## Expert Behavior

The greedy expert should score all 8 phases. For each waiting vehicle, it should determine the intended connection from the current edge to the next route edge, then add the vehicle's accumulated waiting time to every phase that gives that connection protected green.

This makes the expert choose between a paired phase and a single-slot phase based on total demand. A heavy backlog on one approach can make the protected single-slot phase win, while balanced opposing demand can still favor the original paired phases.

Phase 7 on 3-way junctions has no protected movements, so it will normally receive a score of zero. It can still be selected by learned policies because it is part of the fixed action space, but the expert will not prefer it unless all scored phases are also zero and tie-breaking reaches it.

## Environment Behavior

The environment continues to accept one integer target phase per junction. It should support actions `0..7` everywhere.

Switch timing remains unchanged:

- 3 seconds yellow from the current phase.
- 2 seconds all-red.
- 10 seconds target green.
- 15-second decision interval.

If a 3-way junction receives phase 7, its target green state is all red for the green portion of the interval. This is deliberate under this design.

## Testing And Validation

Update tests and verification scripts for the 45-dimensional observation vector and 8-action phase space.

Focused checks:

- `build_junction_info` creates eight green states, eight yellow states, and eight served-lane sets.
- Phases 0-3 match the previous mapping.
- Phases 4-7 open exactly one slot on 4-way junctions.
- Phase 7 is all red for 3-way junctions.
- Expert scoring considers multiple protected phases per connection.
- `verify_env.py` passes with the new observation shape.

## Non-Goals

- No dynamic action mask.
- No generated conflict-compatible phase search.
- No checkpoint migration for existing 4-action models.
- No change to movement feature layout or reward definition.
