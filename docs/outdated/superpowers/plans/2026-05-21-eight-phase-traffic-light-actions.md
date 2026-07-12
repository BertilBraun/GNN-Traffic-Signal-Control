# Eight-Phase Traffic-Light Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the traffic-light controller from four canonical phases to eight phases, adding protected single-approach clearance phases while keeping a fixed action space for 3-way and 4-way junctions.

**Architecture:** Add a shared phase schema module that owns phase/action counts, observation indices, and slot-to-phase mappings. Update runtime junction parsing, SUMO network TLL generation, observations, model defaults, expert scoring, training/eval metrics, and verification scripts to consume the shared constants. Phase 7 is a normal slot-3 clearance phase on 4-way junctions and an all-red pause on 3-way junctions because slot 3 has no controlled links.

**Tech Stack:** Python, SUMO/sumolib/TraCI, NumPy, PyTorch, PyTorch Geometric, pytest-style smoke tests via existing scripts.

---

## File Structure

- Create `src/environment/phase_schema.py`: shared constants and movement-to-phase mapping helpers.
- Modify `src/environment/junction_info.py`: build 8 phase states and map each connection to all protected phases.
- Modify `scripts/build_network.py`: use the same phase schema when writing `.tll.xml` programs.
- Modify `src/environment/sumo_env.py`: change observation length and phase one-hot slice from 4 to 8.
- Modify `src/utils/graph_builder.py`: change normalizer/input feature dimension from 41 to 45.
- Modify `src/model/gat_policy.py`: default `node_dim=45`, `n_classes=8`.
- Modify `src/environment/expert.py`: score 8 phases using `conn_to_phases`.
- Modify `src/training/eval_episode.py`, `src/training/ppo.py`, `src/training/imitation.py`, `scripts/run_grid.py`, `scripts/renormalize.py`, and checkpoint loaders: replace 4-phase assumptions with shared constants where they affect action counts, histograms, burn-in, docs, or model construction.
- Modify `scripts/verify_env.py`: expect 45-dim observations, 8 one-hot phase features, and 8-action expert coverage reporting.

---

### Task 1: Add Shared Phase Schema

**Files:**
- Create: `src/environment/phase_schema.py`
- Modify: `src/environment/__init__.py`

- [ ] **Step 1: Create the phase schema module**

Add `src/environment/phase_schema.py`:

```python
"""Shared phase/action schema for canonical traffic-light control."""
from __future__ import annotations

NUM_APPROACH_SLOTS = 4
MOVEMENT_FEATURES_PER_SLOT = 9
MOVEMENT_FEATURE_DIM = NUM_APPROACH_SLOTS * MOVEMENT_FEATURES_PER_SLOT
NUM_PHASES = 8
PHASE_FEATURE_START = MOVEMENT_FEATURE_DIM
ELAPSED_FEATURE_INDEX = PHASE_FEATURE_START + NUM_PHASES
OBS_DIM = ELAPSED_FEATURE_INDEX + 1

MOVEMENT_ORDER = ("l", "s", "r")

# Signal char: "G" = protected green, "g" = permissive green.
SLOT_DIR_PHASES: dict[tuple[int, str], list[tuple[int, str]]] = {
    (0, "s"): [(0, "G"), (4, "G")],
    (0, "r"): [(0, "G"), (3, "g"), (4, "G")],
    (0, "l"): [(1, "G"), (4, "G")],
    (1, "s"): [(2, "G"), (5, "G")],
    (1, "r"): [(2, "G"), (1, "g"), (5, "G")],
    (1, "l"): [(3, "G"), (5, "G")],
    (2, "s"): [(0, "G"), (6, "G")],
    (2, "r"): [(0, "G"), (3, "g"), (6, "G")],
    (2, "l"): [(1, "G"), (6, "G")],
    (3, "s"): [(2, "G"), (7, "G")],
    (3, "r"): [(2, "G"), (1, "g"), (7, "G")],
    (3, "l"): [(3, "G"), (7, "G")],
}


def phase_indices() -> range:
    """Return all fixed phase indices."""
    return range(NUM_PHASES)
```

- [ ] **Step 2: Export constants for local imports**

Update `src/environment/__init__.py` to:

```python
from .sumo_env import TrafficEnv
from .expert import GreedyExpert
from .phase_schema import NUM_PHASES, OBS_DIM
```

- [ ] **Step 3: Verify imports**

Run:

```bash
python -c "from src.environment.phase_schema import NUM_PHASES, OBS_DIM, SLOT_DIR_PHASES; print(NUM_PHASES, OBS_DIM, SLOT_DIR_PHASES[(0, 'l')])"
```

Expected output contains:

```text
8 45 [(1, 'G'), (4, 'G')]
```

- [ ] **Step 4: Commit**

```bash
git add src/environment/phase_schema.py src/environment/__init__.py
git commit -m "Add shared traffic phase schema"
```

---

### Task 2: Update Runtime Junction Phase Construction

**Files:**
- Modify: `src/environment/junction_info.py`

- [ ] **Step 1: Replace local phase constants**

Import the shared schema near the top:

```python
from .phase_schema import NUM_PHASES, SLOT_DIR_PHASES
```

Remove the local `_SLOT_DIR_PHASES` dictionary. Update comments/docstrings from "4 canonical" to "8 canonical" where they describe phase count.

- [ ] **Step 2: Build eight phase strings**

In `_build_phase_strings`, replace:

```python
phase_states = [['r'] * n_links for _ in range(4)]
```

with:

```python
phase_states = [["r"] * n_links for _ in range(NUM_PHASES)]
```

Replace `_SLOT_DIR_PHASES.get(key, [])` with `SLOT_DIR_PHASES.get(key, [])`.

- [ ] **Step 3: Build eight served-lane sets**

In `_build_phase_served_lanes`, replace:

```python
served: list[set] = [set() for _ in range(4)]
```

with:

```python
served: list[set] = [set() for _ in range(len(green_states))]
```

- [ ] **Step 4: Replace `conn_to_phase` with `conn_to_phases`**

Update the dataclass field:

```python
# (from_edge_id, to_edge_id) -> phase indices giving protected green ("G").
conn_to_phases: dict = field(default_factory=dict)
```

Replace the connection mapping block with:

```python
conn_to_phases: dict[tuple[str, str], list[int]] = {}
for in_edge in node.getIncoming():
    for out_edge in in_edge.getOutgoing():
        for conn in in_edge.getConnections(out_edge):
            tl_idx = conn.getTLLinkIndex()
            if tl_idx < 0:
                continue
            key = (in_edge.getID(), out_edge.getID())
            phases = conn_to_phases.setdefault(key, [])
            for ph_idx, state in enumerate(green_states):
                if tl_idx < len(state) and state[tl_idx] == "G" and ph_idx not in phases:
                    phases.append(ph_idx)
```

Return it as:

```python
conn_to_phases=conn_to_phases,
```

- [ ] **Step 5: Run a syntax/import check**

Run:

```bash
python -m py_compile src/environment/junction_info.py
```

Expected: command exits with code 0.

- [ ] **Step 6: Commit**

```bash
git add src/environment/junction_info.py
git commit -m "Build eight runtime signal phases"
```

---

### Task 3: Update Network TLL Builder

**Files:**
- Modify: `scripts/build_network.py`

- [ ] **Step 1: Import shared phase constants**

Add after `import sumolib`:

```python
from src.environment.phase_schema import NUM_PHASES, SLOT_DIR_PHASES
```

- [ ] **Step 2: Replace the local phase map**

Delete `_PHASE_MAP`. In `_build_phase_strings`, replace:

```python
phase_states = [['r'] * n_links for _ in range(4)]
```

with:

```python
phase_states = [["r"] * n_links for _ in range(NUM_PHASES)]
```

Replace:

```python
for ph, ch in _PHASE_MAP.get((slot_idx, direction.lower()), []):
```

with:

```python
for ph, ch in SLOT_DIR_PHASES.get((slot_idx, direction.lower()), []):
```

- [ ] **Step 3: Update script text**

Change the top pipeline comment from:

```python
6. TLL         — canonical 4-phase signal programs
```

to:

```python
6. TLL         — canonical 8-phase signal programs
```

- [ ] **Step 4: Run a syntax/import check**

Run:

```bash
python -m py_compile scripts/build_network.py
```

Expected: command exits with code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_network.py
git commit -m "Use eight-phase schema in network builder"
```

---

### Task 4: Update Observation Shape And Graph Input

**Files:**
- Modify: `src/environment/sumo_env.py`
- Modify: `src/utils/graph_builder.py`

- [ ] **Step 1: Update `sumo_env.py` constants and docs**

Import:

```python
from .phase_schema import (
    ELAPSED_FEATURE_INDEX,
    MOVEMENT_ORDER,
    NUM_PHASES,
    OBS_DIM,
    PHASE_FEATURE_START,
)
```

Remove the local `MOVEMENT_ORDER = ('l', 's', 'r')`.

Update the module docstring examples from `np.ndarray(41,)` to `np.ndarray(45,)`, actions from `phase 0-3` to `phase 0-7`, and the phase one-hot section to:

```text
[36..43] current phase one-hot (8 dims)
[44]     elapsed time in phase, normalised to [0, 1]
```

- [ ] **Step 2: Allocate 45-dimensional observations**

In `_observe_junction`, replace:

```python
feat = np.zeros(41, dtype=np.float32)
```

with:

```python
feat = np.zeros(OBS_DIM, dtype=np.float32)
```

Replace:

```python
feat[36 + phase] = 1.0
```

with:

```python
if 0 <= phase < NUM_PHASES:
    feat[PHASE_FEATURE_START + phase] = 1.0
```

Replace:

```python
feat[40] = min(1.0, elapsed / MAX_PHASE_HOLD_S)
```

with:

```python
feat[ELAPSED_FEATURE_INDEX] = min(1.0, elapsed / MAX_PHASE_HOLD_S)
```

- [ ] **Step 3: Update GraphBuilder dimensions**

In `src/utils/graph_builder.py`, import:

```python
from src.environment.phase_schema import OBS_DIM
```

Replace `RunningNormalizer(dim=41)` with:

```python
self.normalizer = RunningNormalizer(dim=OBS_DIM)
```

Update docstrings/comments mentioning 41 to use 45 or `OBS_DIM`.

- [ ] **Step 4: Run syntax checks**

Run:

```bash
python -m py_compile src/environment/sumo_env.py src/utils/graph_builder.py
```

Expected: command exits with code 0.

- [ ] **Step 5: Commit**

```bash
git add src/environment/sumo_env.py src/utils/graph_builder.py
git commit -m "Expand observations for eight phases"
```

---

### Task 5: Update Model And Expert Action Counts

**Files:**
- Modify: `src/model/gat_policy.py`
- Modify: `src/environment/expert.py`

- [ ] **Step 1: Update model defaults**

In `src/model/gat_policy.py`, import:

```python
from src.environment.phase_schema import NUM_PHASES, OBS_DIM
```

Change constructor defaults:

```python
node_dim: int = OBS_DIM,
n_classes: int = NUM_PHASES,
```

Update docstrings to say 45 input features and 8 phase classes.

- [ ] **Step 2: Update expert scoring**

In `src/environment/expert.py`, import:

```python
from .phase_schema import NUM_PHASES, phase_indices
```

Replace:

```python
scores = [0.0] * 4
```

with:

```python
scores = [0.0] * NUM_PHASES
```

Replace:

```python
phase = ji.conn_to_phase.get((route[ridx], route[ridx + 1]))
if phase is not None:
    scores[phase] += wait
```

with:

```python
for phase in ji.conn_to_phases.get((route[ridx], route[ridx + 1]), []):
    scores[phase] += wait
```

Replace every `range(4)` used to rank phases with `phase_indices()`.

- [ ] **Step 3: Run syntax checks**

Run:

```bash
python -m py_compile src/model/gat_policy.py src/environment/expert.py
```

Expected: command exits with code 0.

- [ ] **Step 4: Commit**

```bash
git add src/model/gat_policy.py src/environment/expert.py
git commit -m "Train and score eight traffic phases"
```

---

### Task 6: Update Training, Evaluation, And Utility Phase Counts

**Files:**
- Modify: `src/training/eval_episode.py`
- Modify: `src/training/ppo.py`
- Modify: `src/training/imitation.py`
- Modify: `scripts/run_grid.py`
- Modify: `scripts/renormalize.py`

- [ ] **Step 1: Update eval phase histograms**

In `src/training/eval_episode.py`, import:

```python
from src.environment.phase_schema import NUM_PHASES, phase_indices
```

Replace:

```python
junc_phase_cnt  = {jid: [0, 0, 0, 0] for jid in junction_ids}
```

with:

```python
junc_phase_cnt = {jid: [0 for _ in phase_indices()] for jid in junction_ids}
```

Replace:

```python
for ph in range(4)
```

with:

```python
for ph in phase_indices()
```

Update the dataclass docstring from `[ph0, ph1, ph2, ph3]` to `[ph0, ..., ph7]`.

- [ ] **Step 2: Update PPO fixed-time burn-in**

In `src/training/ppo.py`, import:

```python
from src.environment.phase_schema import NUM_PHASES
```

Replace:

```python
phase = (step_in_episode // BURN_IN_CYCLE_LENGTH) % 4
```

with:

```python
phase = (step_in_episode // BURN_IN_CYCLE_LENGTH) % NUM_PHASES
```

Update the burn-in docstring to describe `0→1→...→7→0` and a 240-second cycle at two decision intervals per phase.

- [ ] **Step 3: Update imitation comments and checkpoint loading**

In `src/training/imitation.py`, run:

```bash
rg -n "0-3|\\(N, 4\\)|\\(T\\*N, 41\\)|41" src/training/imitation.py
```

Apply these exact text replacements where they appear in comments or docstrings:

```text
phases 0-3 -> phases 0-7
(N, 4) -> (N, 8)
(T*N, 41) -> (T*N, 45)
41-dim -> 45-dim
```

If `load_checkpoint` constructs `GATPolicy()` with defaults, leave the call unchanged because Task 5 updates the defaults.

- [ ] **Step 4: Update run and renormalize helpers**

In `scripts/renormalize.py`, import `NUM_PHASES` and replace:

```python
phase = (step_count // _FIXED_TIME_CYCLE) % 4
```

with:

```python
phase = (step_count // _FIXED_TIME_CYCLE) % NUM_PHASES
```

In `scripts/run_grid.py`, update help/docstrings/comments that describe four phases or action shape. Model argmax needs no logic change after Task 5.

- [ ] **Step 5: Run syntax checks**

Run:

```bash
python -m py_compile src/training/eval_episode.py src/training/ppo.py src/training/imitation.py scripts/run_grid.py scripts/renormalize.py
```

Expected: command exits with code 0.

- [ ] **Step 6: Commit**

```bash
git add src/training/eval_episode.py src/training/ppo.py src/training/imitation.py scripts/run_grid.py scripts/renormalize.py
git commit -m "Update training utilities for eight phases"
```

---

### Task 7: Update Environment Verification

**Files:**
- Modify: `scripts/verify_env.py`

- [ ] **Step 1: Import shared dimensions**

Add:

```python
from src.environment.phase_schema import (
    ELAPSED_FEATURE_INDEX,
    NUM_PHASES,
    OBS_DIM,
    PHASE_FEATURE_START,
    phase_indices,
)
```

- [ ] **Step 2: Update feature names**

Replace the hard-coded phase name loop:

```python
for _p in range(4):
    _FEAT_NAMES.append(f'phase{_p}_oh')
_FEAT_NAMES.append('elapsed')
assert len(_FEAT_NAMES) == 41
```

with:

```python
for _p in phase_indices():
    _FEAT_NAMES.append(f'phase{_p}_oh')
_FEAT_NAMES.append('elapsed')
assert len(_FEAT_NAMES) == OBS_DIM
```

- [ ] **Step 3: Update shape checks and slices**

Replace shape checks `(41,)` and graph shape `(n, 41)` with `(OBS_DIM,)` and `(n, OBS_DIM)`.

Replace:

```python
if 36 <= i < 40:
    ...
    ph_slice = mat[row_idx, 36:40]
...
if i == 40:
...
ph_sums = mat[:, 36:40].sum(axis=1)
```

with:

```python
if PHASE_FEATURE_START <= i < ELAPSED_FEATURE_INDEX:
    ...
    ph_slice = mat[row_idx, PHASE_FEATURE_START:ELAPSED_FEATURE_INDEX]
...
if i == ELAPSED_FEATURE_INDEX:
...
ph_sums = mat[:, PHASE_FEATURE_START:ELAPSED_FEATURE_INDEX].sum(axis=1)
```

- [ ] **Step 4: Update expert phase coverage**

Replace:

```python
phase_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
```

with:

```python
phase_counts: dict[int, int] = {p: 0 for p in phase_indices()}
```

Change the pass/fail rule to avoid requiring all eight phases in short low-demand runs:

```python
used = [p for p, c in phase_counts.items() if c > 0]
if all(p in used for p in (0, 1, 2, 3)) and any(p in used for p in (4, 5, 6, 7)):
    _ok('base phases and at least one single-slot phase selected')
else:
    _warn(f'limited expert coverage in short episodes; used phases: {used}')
```

Keep the switch-frequency check unchanged.

- [ ] **Step 5: Run syntax check**

Run:

```bash
python -m py_compile scripts/verify_env.py
```

Expected: command exits with code 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_env.py
git commit -m "Verify eight-phase environment features"
```

---

### Task 8: End-To-End Verification

**Files:**
- No planned source edits unless verification exposes a defect in the files changed above.

- [ ] **Step 1: Search for stale hard-coded dimensions**

Run:

```bash
rg -n "41|phase 0-3|phases 0-3|range\\(4\\)|\\[0, 0, 0, 0\\]|n_classes: int = 4|node_dim:.*41|% 4" src scripts tests -S
```

Expected: remaining matches are either unrelated numeric values, historical docs outside runtime code, or code that loops over four approach slots rather than four phases.

- [ ] **Step 2: Run environment verification**

Run:

```bash
python scripts/verify_env.py --episodes 1
```

Expected: startup, observation shape, observation bounds, ordering, and reward sanity pass. Expert coverage may warn about limited short-run coverage but should not fail solely because phase 7 is unused.

- [ ] **Step 3: Run a short model shape smoke test**

Run:

```bash
python - <<'PY'
import torch
from torch_geometric.data import Data
from src.environment.phase_schema import OBS_DIM, NUM_PHASES
from src.model.gat_policy import GATPolicy

model = GATPolicy()
data = Data(
    x=torch.zeros((3, OBS_DIM), dtype=torch.float32),
    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    edge_attr=torch.zeros((2, 3), dtype=torch.float32),
)
logits, values = model.forward_actor_critic(data)
print(tuple(logits.shape), tuple(values.shape), NUM_PHASES)
PY
```

Expected output:

```text
(3, 8) (3,) 8
```

- [ ] **Step 4: Commit verification fixes if needed**

If Step 1-3 expose a source defect, fix the defect and commit the exact touched files:

```bash
git add <fixed-files>
git commit -m "Fix eight-phase verification issues"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review Notes

- Spec coverage: tasks cover shared 8-phase mapping, unchanged phases 0-3, added protected single-slot phases 4-7, fixed no-mask action space, 45-dim observations, 8-class model output, expert scoring for multiple protected phases, phase 7 all-red behavior on 3-way junctions, verification updates, and no checkpoint migration.
- Placeholder scan: no task uses deferred placeholders; every code-changing step includes concrete snippets or exact replacements.
- Type consistency: the plan consistently uses `NUM_PHASES`, `OBS_DIM`, `PHASE_FEATURE_START`, `ELAPSED_FEATURE_INDEX`, `SLOT_DIR_PHASES`, and `conn_to_phases`.
