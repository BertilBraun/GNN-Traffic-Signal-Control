# City-Building Pipeline

The city pipeline turns a reproducible OSM source into a movement-safe SUMO scenario. Raw OSM is not assumed to be directly suitable for training: network construction, topology shaping, signal synthesis, routing, and demand calibration are part of the scenario definition.

## Saved inputs and generated outputs

Each city directory contains the inputs needed to rebuild it:

```text
configs/<city>/
  <city>.build.yaml    build, demand, and verification recipe
  <city>.osm           cached OSM source
  <city>.prune.json    replayable manual shaping decisions
```

The workbench generates the SUMO network, routes, signal program, additional file, SUMO config, inspection summary, and movement-graph reports. Generated files may be replaced by rerunning the saved recipe; the OSM/build/prune inputs are the reproducibility boundary.

## Network workbench

For an existing city recipe, the complete interactive pipeline is:

```powershell
uv run python scripts\network_workbench.py `
  --build-file configs\freiburg_altstadt\freiburg_altstadt.build.yaml `
  all
```

`all` performs the following sequence:

1. Build from the cached OSM source and any existing prune recipe.
2. Open the pruning editor and wait for pruning to finish.
3. Rebuild from the saved prune JSON.
4. Inspect movement extraction and signal compatibility.
5. Generate configured graph and detector visualizations.
6. Write the build summary.
7. Open SUMO-GUI for demand calibration when enabled by the recipe.

Individual workbench commands are available for `fetch`, `build-initial`, `prune`, `rebuild`, `inspect`, `visualize`, `run-gui`, `calibrate-demand`, and `evaluate`.

## Starting a new city

Create a saved build recipe and initial network from a bounding box:

```powershell
uv run python scripts\network_workbench.py `
  --name munich_small `
  --bbox 48.147,11.568,48.155,11.581 `
  --out-dir configs\munich_small `
  build-initial
```

Then review and save topology edits:

```powershell
uv run python scripts\network_workbench.py `
  --build-file configs\munich_small\munich_small.build.yaml `
  --open `
  prune
```

Finish with `rebuild`, `inspect`, `visualize`, and `calibrate-demand`, or run `all` after the recipe exists.

## Shaping and pruning constraints

The build should eliminate or isolate topology that cannot produce a reliable movement graph or feasible vehicle routes. Typical pruning targets include:

- malformed or unsupported junction geometry;
- infeasible connectors and route-validity failures;
- small disconnected or ambiguous components;
- duplicate, over-joined, or misleading signal clusters;
- road fragments that create implausible origin-destination shortcuts;
- signalized junctions whose controlled links cannot be mapped to supported movements.

Corridor contraction through unsignalized junctions is allowed only when continuation is unambiguous. A branch may select a unique straight continuation when the topology supports one; otherwise contraction stops.

The pruning editor saves decisions to `<city>.prune.json`. A rebuild replays those decisions against the cached OSM source, preventing undocumented manual edits to generated network XML.

## Signal and movement synthesis

Only imported or deliberately promoted traffic lights are candidates for control. The builder extracts legal controlled turns, groups compatible turns into selectable phases, and generates deterministic yellow transitions. A controllable junction must have a consistent mapping among SUMO controlled links, movement nodes, and synthesized phase incidence.

Junctions that cannot satisfy these requirements remain pass-through, are excluded from policy decisions, or are removed during shaping. The policy never invents signal states for an unsupported junction.

## Demand and initial occupancy

The city recipe specifies a base demand rate and a route-count limit. The builder samples valid origin-destination routes over the shaped topology and writes route flows. Runtime demand scale multiplies the base flows without rebuilding the network.

Training can additionally sample a target initial occupancy and place vehicles at safe random positions on feasible routes. This exposes the controller to congested initial states rather than always starting from an empty network. Demand scale and initial occupancy are deterministic for a fixed rollout seed.

## Verification

Before a city participates in learning:

```powershell
uv run python scripts\inspect_movement_city.py `
  --cfg configs\<city>\<city>.sumocfg `
  --time-to-teleport -1

uv run python scripts\visualize_movement_graph.py `
  --cfg configs\<city>\<city>.sumocfg `
  --out configs\<city>\reports\movement_graph.html

uv run python scripts\run.py `
  --cfg configs\<city>\<city>.sumocfg `
  --method max-pressure `
  --gui
```

Review route plausibility, lane-group continuity, movement ownership, selectable phase counts, disconnected components, demand level, and gridlock behavior. A successful build is not only one that parses: its traffic and topology must be credible enough to define a useful learning problem.

## Current experiment cities

The iteration-85 experiment uses four rollout cities—Karlsruhe, Mannheim, Stuttgart, and Heidelberg—and one validation city, Freiburg. Their saved recipes and prune files live under `configs/`. Freiburg is omitted from PPO rollout generation by the experiment YAML, not by the city-building pipeline.
