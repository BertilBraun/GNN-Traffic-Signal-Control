# Network Build Pipeline Plan

This plan turns the current city/OSM scripts into one repeatable workflow:

```text
OSM source -> cached raw OSM -> initial SUMO network -> manual prune JSON
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
* `scripts/visualize_network_prune.py` provides the interactive served prune
  editor. In served mode it saves the prune JSON, rebuilds after each click
  or box edit, and reloads the regenerated network. This behavior is now part
  of the expected workflow and should be preserved.
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
  <city>.build.yaml              # saved build inputs for reruns
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
are the saved build inputs and prune JSON.

## Build Inputs

The normal first-run input should be just a city name and bounding box:

```powershell
python scripts\network_workbench.py `
  --name karlsruhe_oststadt `
  --bbox "49.0000,8.4050,49.0230,8.4520" `
  all
```

The command writes `configs/<city>/<city>.build.yaml` so the same build can be
rerun later without retyping the coordinates or demand settings.

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
  gui: true
  gui_steps: 1800
  demand_scale: 4.5
```

The saved build file should let the same command rebuild the same network after
topology pruning or demand adjustment.

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

Manual editing should produce a declarative prune JSON file instead of directly
mutating only the final `.net.xml`.

Initial prune file operations:

```json
{
  "delete_junctions": ["nodeA", "nodeB"],
  "delete_edges": ["edgeC"],
  "keep_junctions": [],
  "notes": [
    {
      "target_id": "nodeA",
      "text": "residential cul-de-sac, no useful signalized path"
    }
  ]
}
```

Semantics:

* Deleting a junction deletes all incident normal edges.
* Deleting an edge removes the road segment between its endpoints, including
  the reverse SUMO edge when the street is represented as two directed edges.
* Internal SUMO edges are never user-selected directly.
* After edits, rebuild connections, traffic-light programs, movement graph, and
  routes from the edited topology.
* If a traffic light loses enough arms to become a pass-through or unsupported
  node, the inspection report should expose that explicitly.

This prune file should be applied before final movement TLL generation and route
generation, otherwise old routes and traffic-light programs can reference
deleted topology.

Acceptance:

* A prune file can delete a residential side component and produce a valid
  `.net.xml`, `.tll.xml`, `.rou.xml`, and `.sumocfg`.
* `inspect_movement_city.py` still runs after pruning.
* Deleting incoming roads changes the affected junction program or demotes it
  through the existing inspection path.
* Re-running the build with the same build file and prune JSON produces the same
  network.

## Prune UI

The first implementation is an HTML workbench because the repository already
has self-contained graph visualizations.

Suggested command:

```powershell
python scripts\visualize_network_prune.py `
  --net configs\karlsruhe_oststadt\karlsruhe_oststadt.net.xml `
  --serve `
  --open
```

UI behavior:

* Render SUMO junctions and normal edges over the current map layout.
* Use click selection for junctions and edges.
* Use box selection for bulk deletion of all junctions and edges under a map
  rectangle; edges are selected by midpoint to avoid grabbing long roads that
  merely cross the box.
* In served mode, save, rebuild, and reload automatically after each click or
  box edit. This is the primary editing mode.
* Provide undo for the last click or box edit within the current editor session;
  undo saves the reverted prune JSON and rebuilds again.
* Show selected object metadata: id, type, incoming/outgoing edge count,
  signalized status, movement count if available, and connected component.
* Provide explicit actions: mark selected junctions and edges for deletion,
  undo the last edit, and rebuild manually when needed.
* In served mode, rebuild directly from the editor; rebuild saves the current
  prune JSON first and reloads the regenerated network on success.
* Visually distinguish signalized junctions, unsignalized junctions, deleted
  objects, and protected objects.
* Never delete immediately from the browser-only view without writing the prune
  JSON.

Implementation notes:

* The served editor runs a tiny local Python HTTP server, accepts `POST /prune`
  to write the prune JSON directly into the config directory, and runs
  `scripts/build_network.py` with that prune JSON.

Acceptance:

* The user can select a visible side area, save deletions, rebuild, and reopen
  the updated view.
* The UI does not expose internal SUMO edges as primary editable objects.
* The saved prune JSON is human-readable and reviewable.

## Orchestrated Command

Add one high-level script after caching and pruning exist:

```powershell
python scripts\network_workbench.py `
  --name karlsruhe_oststadt `
  --bbox "49.0000,8.4050,49.0230,8.4520" `
  all
```

Subcommands:

* `fetch`: resolve OSM source and cache it.
* `build-initial`: run OSM import/netconvert cleanup without manual prune.
* `prune`: open the served prune UI by default, preserving the existing
  auto-save/auto-rebuild/reload loop. The served UI has a finish action that
  saves the prune JSON and lets the workbench continue.
* `rebuild`: non-interactively apply prune JSON and regenerate final network
  artifacts.
* `inspect`: run movement extraction inspection and save report.
* `visualize`: generate movement graph and optional detection HTML.
* `run-gui`: launch `scripts/run.py --gui` with saved demand settings.
* `evaluate`: launch `scripts/eval_policy.py` for baseline seeds.
* `all`: run the interactive workbench sequence: build current network, open
  prune UI and wait for Finish, rebuild from saved prune JSON, inspect, open
  visualization reports, and launch SUMO-GUI by default. It should not run
  evaluation.

The existing `scripts/build_network.py` can remain the low-level builder. The
first orchestrator can call the existing scripts directly. A later cleanup can
split `build_network.py` into a small library module plus a thin CLI once the
workflow shape has stabilized.

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

* `network_workbench.py rebuild` produces the same core files as
  `build_network.py`.
* `network_workbench.py all` opens the prune UI checkpoint and then produces
  reports and starts SUMO-GUI after the user finishes pruning.
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

* A city's saved build file records the calibrated demand value used for later
  IL/PPO data collection.
* Demand is not increased to hide a broken topology issue; degenerate corridors
  are fixed by pruning, bbox adjustment, or exclusion.

## Implementation Order

1. Add OSM cache support to the existing build path.
2. Add prune JSON application on plain XML or pre-final `net.xml`.
3. Add the first prune visualization/editor with served auto-rebuild editing.
4. Add saved build-file read/write support.
5. Add `scripts/network_workbench.py` subcommands around existing tools.
6. Extract city-build functions from `scripts/build_network.py` into a reusable
   module while preserving the current CLI behavior.
7. Add saved reports and build summary JSON.
8. Add documentation examples for the five current city candidates.

## Validation Checklist

For each edited city network:

```powershell
python scripts\network_workbench.py --build-file configs\<city>\<city>.build.yaml all

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
