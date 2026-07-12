# Movement Max-Pressure Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first movement-aware controller checkpoint: SUMO phase extraction plus a max-pressure phase selector.

**Architecture:** Move the previous fixed 8-phase source modules under `src/legacy/` and create a new `src/movement/` package. The new package starts with pure, testable extraction and scoring code before any TraCI environment integration.

**Tech Stack:** Python 3, pytest, SUMO/TraCI-compatible data shapes, dataclasses.

---

## File Structure

- Move: `src/environment/` -> `src/legacy/environment/`
- Move: `src/model/` -> `src/legacy/model/`
- Move: `src/training/` -> `src/legacy/training/`
- Move: `src/utils/` -> `src/legacy/utils/`
- Keep: `src/__init__.py`
- Create: `src/legacy/README.md`
- Create: `src/movement/__init__.py`
- Create: `src/movement/schema.py`
- Create: `src/movement/extraction.py`
- Create: `src/movement/max_pressure.py`
- Create: `tests/test_movement_extraction.py`
- Create: `tests/test_max_pressure_policy.py`

## Task 1: Move Legacy Source

**Files:**
- Move source package directories into `src/legacy/`
- Create: `src/legacy/README.md`

- [ ] **Step 1: Move old source directories**

Run PowerShell `Move-Item` for the old source directories into `src/legacy/`.

- [ ] **Step 2: Add legacy README**

Create `src/legacy/README.md` explaining that this is the fixed canonical 8-phase baseline and no longer the main implementation path.

- [ ] **Step 3: Inspect moved tree**

Run: `Get-ChildItem -Path src -Force`

Expected: `legacy` exists and old source directories are under it.

## Task 2: Movement Extraction Schema

**Files:**
- Create: `tests/test_movement_extraction.py`
- Create: `src/movement/schema.py`
- Create: `src/movement/extraction.py`

- [ ] **Step 1: Write failing extraction tests**

Test that green phases are selected, yellow/all-red/empty phases are filtered, and phase movement indices match controlled-link positions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_movement_extraction.py -q`

Expected: FAIL because `src.movement` does not exist yet.

- [ ] **Step 3: Implement minimal schema and extraction**

Define `ControlledMovement`, `SelectablePhase`, `TrafficLightProgram`, `extract_traffic_light_program`, and `is_selectable_green_state`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_movement_extraction.py -q`

Expected: PASS.

## Task 3: Max-Pressure Phase Selection

**Files:**
- Create: `tests/test_max_pressure_policy.py`
- Create: `src/movement/max_pressure.py`

- [ ] **Step 1: Write failing max-pressure tests**

Test that phase pressure is aggregated over active movements and that the highest-scoring phase is selected with deterministic tie-breaking.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_max_pressure_policy.py -q`

Expected: FAIL because `src.movement.max_pressure` does not exist yet.

- [ ] **Step 3: Implement minimal max-pressure selector**

Define `score_phase_pressure`, `score_program_phases`, and `select_max_pressure_phase`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_max_pressure_policy.py -q`

Expected: PASS.

## Task 4: Verification

**Files:**
- New movement tests
- Legacy tree

- [ ] **Step 1: Run targeted movement tests**

Run: `pytest tests/test_movement_extraction.py tests/test_max_pressure_policy.py -q`

Expected: PASS.

- [ ] **Step 2: Check repository status**

Run: `git status --short`

Expected: moved legacy files plus new movement package and tests.

## Deferred After This Checkpoint

- TraCI-backed `MovementTrafficEnv`.
- GUI runner for visual verification.
- Road-link graph compression.
- PPO or MAPPO integration.
- Full suite repair after old imports are intentionally moved.
