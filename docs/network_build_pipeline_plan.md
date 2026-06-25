# Network Build Pipeline Plan

This plan turns the current city/OSM scripts into one repeatable workflow:

```text
OSM source -> cached raw OSM -> initial SUMO network -> manual prune recipe
-> rebuilt SUMO network -> movement traffic-light programs -> routes/config
-> inspection reports -> HTML visualization -> optional SUMO-GUI demand check
```

The goal is not to remove manual judgment. The goal is to make every manual
decision explicit, replayable, and attached to the generated network artifact.

All implementation work for this pipeline should follow `CODING_STANDARDS.md`.
In particular, new serialized workflow state should use typed models rather than
raw dictionaries, fixed option sets should use enums, and formatting should pass
`ruff format` plus `ruff check --fix` before each commit.

## Current Building Blocks

Existing tools already cover most of the low-level work:

* `scripts/build_network.py` downloads or loads OSM, runs `netconvert`, cleans
  topology, writes movement-safe `.tll.xml`, writes city routes, and writes
  `.sumocfg`.
* `scripts/inspect_movement_city.py` starts the movement runtime, extracts
  selectable programs, builds the movement graph, and prints graph health.
* `scripts/visualize_movement_graph.py` writes an interactive HTML movement
  graph.
* `scripts/visualize_movement_detection.py` writes an HTML view of detector-like
  observation windows and sampled traffic state.
* `scripts/run.py --gui` runs a policy in SUMO-GUI and is the current manual
  demand-calibration tool.
* `scripts/eval_policy.py` runs headless baseline evaluation once a config is
  stable.

The missing layer is a coherent network-workbench command that coordinates
these tools and persists the intermediate decisions.

## Artifact Layout

Each city build should have a stable working directory:

```text
configs/<city>/
  <city>.build.yaml              # replayable build recipe
  <city>.osm                     # cached raw OSM source
  <city>.prune.json              # manual topology edits
  <city>.net.xml                 # final SUMO network
  <city>.tll.xml                 # movement-safe traffic-light programs
  <city>.rou.xml                 # base route flows
  <city>.add.xml                 # additional file
  <city>.sumocfg                 # runnable SUMO config
  reports/
    inspection.txt
    movement_graph.html
    movement_detection.html
    build_summary.json
```

Generated files may stay ignored by Git. The important hand-authored artifacts
are the build recipe and prune recipe.

## Build Recipe

`<city>.build.yaml` should be the canonical input to the future orchestrator.

Suggested fields:

```yaml
name: karlsruhe_oststadt
source:
  bbox: "49.0000,8.4050,49.0230,8.4520"
  cache_policy: reuse
netconvert:
  join_dist: 35
  promote_all_junctions_to_tl: false
demand:
  route_count: 300
  demand_vehicles_per_hour: 900
verification:
  inspect: true
  movement_graph_html: true
  detection_html: false
  gui: false
  gui_steps: 1800
  demand_scale: 4.5
```

The recipe should let the same command rebuild the same network after topology
pruning or demand adjustment.

## OSM Cache

Add a small cache layer before the current Overpass call.

Behavior:

* If `--osm` is passed, use that file directly and do not cache.
* If `--bbox` is passed, compute a cache key from the bbox and the Overpass
  query version.
* Store downloaded OSM under a stable cache path, for example
  `.cache/osm/<sha>.osm`.
* Copy or hardlink the cached file to `configs/<city>/<city>.osm`.
* Default behavior should be `reuse` so repeated builds do not hit Overpass.
* Provide `--refresh-osm` or `cache_policy: refresh` for deliberate re-downloads.

Acceptance:

* Running the same bbox twice uses the cached file on the second run.
* Changing bbox or query version creates a new cache entry.
* The build summary records whether OSM came from cache, download, or explicit
  local file.

## Manual Prune Model

Manual editing should produce a declarative prune recipe instead of directly
mutating only the final `.net.xml`.

Initial prune operations:

```json
{
  "delete_junctions": ["nodeA", "nodeB"],
  "delete_edges": ["edgeC"],
  "keep_junctions": [],
  "notes": {
    "nodeA": "residential cul-de-sac, no useful signalized path"
  }
}
```

Semantics:

* Deleting a junction deletes all incident normal edges.
* Deleting an edge removes that directed road segment.
* Internal SUMO edges are never user-selected directly.
* After edits, rebuild connections, traffic-light programs, movement graph, and
  routes from the edited topology.
* If a traffic light loses enough arms to become a pass-through or unsupported
  node, the inspection report should expose that explicitly.

This recipe should be applied before final movement TLL generation and route
generation, otherwise old routes and traffic-light programs can reference
deleted topology.

Acceptance:

* A recipe can delete a residential side component and produce a valid
  `.net.xml`, `.tll.xml`, `.rou.xml`, and `.sumocfg`.
* `inspect_movement_city.py` still runs after pruning.
* Deleting incoming roads changes the affected junction program or demotes it
  through the existing inspection path.
* Re-running the build with the same recipe produces the same network.

## Prune UI

The first implementation should be an HTML workbench because the repository
already has self-contained graph visualizations.

Suggested command:

```powershell
python scripts\network_workbench.py `
  --recipe configs\karlsruhe_oststadt\karlsruhe_oststadt.build.yaml `
  prune --open
```

UI behavior:

* Render SUMO junctions and normal edges over the current map layout.
* Use click selection for junctions and edges.
* Show selected object metadata: id, type, incoming/outgoing edge count,
  signalized status, movement count if available, and connected component.
* Provide explicit actions: mark selected junctions for deletion, mark selected
  edges for deletion, clear selection, save prune recipe.
* Visually distinguish signalized junctions, unsignalized junctions, deleted
  objects, and protected objects.
* Never delete immediately from the browser-only view without writing a recipe.

Implementation options:

* Fastest path: generate static HTML plus embedded JSON and let the browser
  download an updated `*.prune.json`; then the user places it in the config
  directory.
* Better local path: run a tiny local Python HTTP server that serves the view
  and accepts `POST /prune` to write the recipe directly.

Prefer the local server once editing starts to feel frequent. Static HTML is
acceptable for the first proof of concept.

Acceptance:

* The user can select a visible side area, save deletions, rebuild, and reopen
  the updated view.
* The UI does not expose internal SUMO edges as primary editable objects.
* The saved recipe is human-readable and reviewable.

## Orchestrated Command

Add one high-level script after caching and pruning exist:

```powershell
python scripts\network_workbench.py `
  --recipe configs\karlsruhe_oststadt\karlsruhe_oststadt.build.yaml `
  build
```

Subcommands:

* `fetch`: resolve OSM source and cache it.
* `build-initial`: run OSM import/netconvert cleanup without manual prune.
* `prune`: open or write the prune UI.
* `rebuild`: apply prune recipe and regenerate final network artifacts.
* `inspect`: run movement extraction inspection and save report.
* `visualize`: generate movement graph and optional detection HTML.
* `run-gui`: launch `scripts/run.py --gui` with recipe demand settings.
* `evaluate`: launch `scripts/eval_policy.py` for baseline seeds.
* `all`: run the non-interactive sequence through inspection/visualization.

The existing `scripts/build_network.py` can remain the low-level builder. The
new script should call into extracted functions from `build_network.py`, or
`build_network.py` should be split into a small library module plus a thin CLI.

Recommended module split:

```text
src/movement/city_build/cache.py
src/movement/city_build/recipe.py
src/movement/city_build/build.py
src/movement/city_build/prune.py
src/movement/city_build/reports.py
scripts/network_workbench.py
```

Acceptance:

* `network_workbench.py all` produces the same core files as
  `build_network.py` plus reports.
* Each subcommand can be run independently during manual iteration.
* The command prints the next useful manual step when it stops.

## Manual Demand Calibration Loop

Demand calibration should stay manual at first, but the loop should be explicit.

Workflow:

1. Build or rebuild the network.
2. Run inspection and movement graph visualization.
3. Launch GUI with a candidate `demand_scale`.
4. Simulate roughly `1000..2000` steps.
5. Record a simple verdict in the build summary:
   `too_low`, `usable`, `too_high`, or `topology_problem`.
6. Adjust `demand.demand_vehicles_per_hour` or verification
   `demand_scale`, then rebuild or rerun as appropriate.

Important distinction:

* `demand_vehicles_per_hour` changes the base route file.
* `--demand-scale` changes runtime scaling of that route file.

Use runtime scaling for quick visual checks. Change the base route demand once
the city has a plausible stable range.

Acceptance:

* A city recipe records the calibrated demand value used for later IL/PPO data
  collection.
* Demand is not increased to hide a broken topology issue; degenerate corridors
  are fixed by pruning, bbox adjustment, or exclusion.

## Implementation Order

1. Add OSM cache support to the existing build path.
2. Extract city-build functions from `scripts/build_network.py` into a reusable
   module while preserving the current CLI behavior.
3. Add build recipe read/write support.
4. Add prune recipe application on plain XML or pre-final `net.xml`.
5. Add the first prune visualization/editor.
6. Add `scripts/network_workbench.py` subcommands around existing tools.
7. Add saved reports and build summary JSON.
8. Add documentation examples for the five current city candidates.

## Validation Checklist

For each edited city network:

```powershell
python scripts\network_workbench.py --recipe configs\<city>\<city>.build.yaml all

python scripts\inspect_movement_city.py `
  --cfg configs\<city>\<city>.sumocfg `
  --time-to-teleport -1

python scripts\visualize_movement_graph.py `
  --cfg configs\<city>\<city>.sumocfg `
  --out configs\<city>\reports\movement_graph.html `
  --open

python scripts\run.py `
  --cfg configs\<city>\<city>.sumocfg `
  --method max-pressure `
  --gui `
  --demand-scale <candidate> `
  --time-to-teleport -1
```

A network is ready for training only when:

* OSM source is cached or explicitly local.
* Any manual pruning is replayable from `*.prune.json`.
* Movement inspection is clean or has documented acceptable skips.
* Movement graph HTML matches the intended controlled area.
* SUMO-GUI demand looks plausible for `1000..2000` steps.
* Baseline headless evaluation runs without teleport/gridlock artifacts.
