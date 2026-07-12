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

### From OSM geometry to controlled links

Phases are not inferred directly from OSM tags. `netconvert` first turns the shaped OSM road geometry, lane permissions, and junction priorities into a SUMO network with lane-to-lane connections, traffic-light link indices, junction request indices, and a request-conflict (`foes`) matrix. The city builder converts each signal-controlled connection into a link specification containing:

- its traffic-light link index and SUMO request index;
- its incoming and outgoing lane;
- its outgoing edge.

Traffic lights with no controlled links are skipped. A junction is also rejected from synthesized control when multiple connections reuse one traffic-light link index, because the current movement representation requires an unambiguous signal-index-to-movement mapping.

### Atomic movement groups

Before searching for phases, controlled links that must activate together are collapsed into atomic groups. Two links belong to the same transitive group when either:

- they share an incoming lane; or
- parallel lanes from the same incoming edge lead to the same outgoing destination (the outgoing lane when known, otherwise the outgoing edge).

This prevents the synthesizer from opening only part of a shared lane or arbitrarily enabling one of several parallel lanes serving the same destination. If links inside an atomic group conflict, the group cannot become a selectable phase component and the junction must be inspected.

### Conflict and compatibility rules

Two atomic groups are compatible only when every cross-group pair passes both rules:

1. SUMO does not mark the two junction requests as foes in either direction.
2. The links do not enter the same outgoing edge from different incoming approaches.

The second rule deliberately treats competing merges as conflicts even when SUMO's request matrix does not. Multiple lanes from the same approach may still enter the same outgoing edge together.

The compatible groups form an undirected graph. Phase synthesis enumerates its **maximal cliques** with the Bron–Kerbosch algorithm. Each clique becomes one selectable green phase: its controlled links are `G`, and all other links are `r`.

“Maximal” does not mean “maximum.” A maximal compatible set is one to which no additional atomic group can be added safely. The builder retains every such set, including smaller protected phases, rather than keeping only the phase with the largest number of movements. Duplicate states are removed and phases are ordered deterministically, with larger sets first. Synthesis stops with an error above 128 maximal phases so an excessively complex junction can be simplified or pruned instead of silently truncated.

For the standalone generated SUMO program, each selectable green is followed by a 3-second yellow and a 2-second all-red phase. During learned control, only the synthesized green states are policy actions; runtime minimum-green and yellow-transition logic mediates switches between them.

Inspect the controlled links, conflicts, and resulting maximal states for any built junction:

```powershell
uv run python scripts\tools\analyze_movement_conflicts.py `
  --net configs\<city>\<city>.net.xml `
  --tls <junction-id> `
  --mode conflict-edge
```

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
