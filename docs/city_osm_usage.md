# City / OSM Movement Network Usage

The OSM build path now targets the movement-based controller, not the legacy
fixed-intersection pipeline. It writes movement-safe traffic-light programs,
city O-D flows, an empty additional file, and a SUMO config.

The planned next step is a replayable network workbench that wraps OSM caching,
manual topology pruning, inspection, visualization, and GUI demand calibration.
See `docs/network_build_pipeline_plan.md` for the detailed implementation plan.

By default, the builder controls only traffic lights imported or guessed from
OSM/netconvert. It does not turn every 3+-arm city junction into a signal. Use
`--promote-all-junctions-to-tl` only for experiments where you intentionally
want every intersection controlled like a generated grid.

## Build A City Config

```powershell
python scripts\build_network.py `
  --bbox 48.147,11.568,48.155,11.581 `
  --out-dir configs\munich_small `
  --name munich_small `
  --route-count 300 `
  --demand-vehicles-per-hour 900
```

Karlsruhe-Oststadt inspection config:

```powershell
python scripts\build_network.py `
  --bbox 49.0000,8.4050,49.0230,8.4520 `
  --out-dir configs\karlsruhe_oststadt `
  --name karlsruhe_oststadt `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900
```

Generated city configs are reproducible artifacts. Prefer keeping the command
in docs and regenerating them locally instead of committing generated `.net.xml`,
`.rou.xml`, `.sumocfg`, HTML reports, or SUMO transient outputs.

## First-Pass Multi-City Training Set

Use this first city split for movement-score IL collection and transfer checks:

| split | city config | bbox (S,W,N,E) | rationale | expected risk |
| --- | --- | --- | --- | --- |
| train | `karlsruhe_oststadt` | `49.0000,8.4050,49.0230,8.4520` | Personally relevant Karlsruhe east-side area; moderate size with a mix of arterial, campus, and neighborhood streets. | Lower signal density than larger cores; keep as anchor/reference rather than the hardest training case. |
| train | `mannheim_innenstadt` | `49.4780,8.4550,49.4980,8.4930` | Dense signalized Mannheim core around the Quadrate plus approaches; useful for many control targets. | Many skipped/unsupported imported traffic lights; inspect secondary components before relying on all movements. |
| train | `stuttgart_mitte` | `48.7645,9.1580,48.7870,9.1975` | Dense, irregular central Stuttgart basin with many signalized movements and non-grid topology. | Larger movement count; runtime cost higher than Karlsruhe/Heidelberg. |
| train | `heidelberg_bergheim` | `49.3980,8.6720,49.4200,8.7100` | Heidelberg Bergheim/Weststadt/central approaches; compact, irregular, and easier to inspect manually. | Fewer controllable lights than Mannheim/Stuttgart; useful as a stable mid-size training city. |
| held-out eval | `freiburg_altstadt` | `47.9860,7.8290,48.0100,7.8690` | Freiburg Altstadt/Stuehlinger/Wiehre approaches; irregular old-center topology and good transfer target. | Moderate component fragmentation in peripheral lane groups; inspect GUI demand before transfer claims. |

Frankfurt-Nordend/Bornheim was tried as a stronger dense candidate, but the
initial bbox produced SUMO route-validity failures in generated demand. Revisit
it after the first five-city loop is stable.

Build commands:

```powershell
python scripts\build_network.py `
  --bbox 49.0000,8.4050,49.0230,8.4520 `
  --out-dir configs\karlsruhe_oststadt `
  --name karlsruhe_oststadt `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900

python scripts\build_network.py `
  --bbox 49.4780,8.4550,49.4980,8.4930 `
  --out-dir configs\mannheim_innenstadt `
  --name mannheim_innenstadt `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900

python scripts\build_network.py `
  --bbox 48.7645,9.1580,48.7870,9.1975 `
  --out-dir configs\stuttgart_mitte `
  --name stuttgart_mitte `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900

python scripts\build_network.py `
  --bbox 49.3980,8.6720,49.4200,8.7100 `
  --out-dir configs\heidelberg_bergheim `
  --name heidelberg_bergheim `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900

python scripts\build_network.py `
  --bbox 47.9860,7.8290,48.0100,7.8690 `
  --out-dir configs\freiburg_altstadt `
  --name freiburg_altstadt `
  --join-dist 35 `
  --route-count 300 `
  --demand-vehicles-per-hour 900
```

Verification commands, one city at a time:

```powershell
python scripts\inspect_movement_city.py `
  --cfg configs\<city_name>\<city_name>.sumocfg `
  --time-to-teleport -1

python scripts\visualize_movement_graph.py `
  --cfg configs\<city_name>\<city_name>.sumocfg `
  --out reports\<city_name>_movement_graph.html

python scripts\run.py `
  --cfg configs\<city_name>\<city_name>.sumocfg `
  --method max-pressure `
  --gui `
  --time-to-teleport -1
```

Initial inspected metrics with passenger-aware route sampling:

| city config | SUMO TLs | selectable TLs | pass-through TLs | lane groups | movements | lane-lane connectors | components | largest component | signalized connector errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `karlsruhe_oststadt` | 49 | 45 | 0 | 711 | 301 | 1147 | 4 | 708 lane groups, 301 movements | 0 |
| `mannheim_innenstadt` | 108 | 86 | 0 | 1004 | 463 | 1476 | 7 | 984 lane groups, 461 movements | 0 |
| `stuttgart_mitte` | 84 | 77 | 0 | 914 | 591 | 1254 | 2 | 913 lane groups, 591 movements | 0 |
| `heidelberg_bergheim` | 64 | 56 | 0 | 744 | 347 | 1049 | 3 | 742 lane groups, 347 movements | 0 |
| `freiburg_altstadt` | 67 | 57 | 0 | 876 | 423 | 1435 | 8 | 866 lane groups, 423 movements | 0 |

Headless smoke check used before manual GUI inspection:

```powershell
python scripts\run.py `
  --cfg configs\<city_name>\<city_name>.sumocfg `
  --method max-pressure `
  --steps 300 `
  --time-to-teleport -1
```

Manual SUMO-GUI checks are still required before data collection: routes should
show plausible internal city trips, not only boundary-to-boundary shortcuts, and
900 vehicles/hour should not produce immediate gridlock. If demand is too light
or overloaded, adjust `--demand-vehicles-per-hour` first, then bbox, before
changing topology code.

Using an existing OSM file:

```powershell
python scripts\build_network.py `
  --osm configs\munich_small\munich_small.osm `
  --out-dir configs\munich_small `
  --name munich_small
```

Using a manual prune recipe:

First build an initial network, then open the prune editor on the generated
`.net.xml`. For quick iteration, use `--serve`; the browser can then save the
recipe directly into the config directory instead of downloading it:

```powershell
python scripts\visualize_network_prune.py `
  --net configs\karlsruhe_oststadt\karlsruhe_oststadt.net.xml `
  --serve `
  --open
```

By default this saves to:

```text
configs\karlsruhe_oststadt\karlsruhe_oststadt.prune.json
```

In `--serve` mode, selecting a junction or road segment saves the current
recipe, runs the rebuild from the local Python server process, and reloads the
map after a successful build. `Undo` restores the previous edit and rebuilds
again. It uses the sibling OSM file when present. The equivalent command is:

```powershell
python scripts\build_network.py `
  --osm configs\karlsruhe_oststadt\karlsruhe_oststadt.osm `
  --out-dir configs\karlsruhe_oststadt `
  --name karlsruhe_oststadt `
  --prune configs\karlsruhe_oststadt\karlsruhe_oststadt.prune.json
```

The first prune recipe format is intentionally replayable JSON:

```json
{
  "delete_junctions": ["junction_id"],
  "delete_edges": ["edge_id"],
  "keep_junctions": [],
  "notes": [
    {
      "target_id": "junction_id",
      "text": "residential side area outside the control target"
    }
  ]
}
```

Deleting a junction also deletes all incident normal edges. The builder applies
the recipe before regenerating connections, movement traffic-light programs,
routes, and the final SUMO config.

Deleting an edge removes the whole road segment between the two endpoints. If
SUMO imported the street as a pair of directed edges, the reverse edge is
removed with it so netconvert does not rebuild one-sided road geometry.

For larger edits, drag a rectangle over the residential side area or dead-end
branch to mark junctions inside the box and road segments whose midpoints are
inside the box. Use the middle mouse button to pan the map. `Undo` reverts the
last click or box edit in the current browser session and rebuilds the network
again.

Be careful with junction deletion: deleting a junction also deletes every
normal road segment incident to that junction. If you only want one road gone,
click that specific edge instead of selecting the junction.

Outputs:

```text
configs/<city>/<city>.net.xml
configs/<city>/<city>.tll.xml
configs/<city>/<city>.rou.xml
configs/<city>/<city>.add.xml
configs/<city>/<city>.sumocfg
```

`<city>.add.xml` is intentionally empty, matching generated grids. Current
movement features compute detector-like observation windows directly from
vehicle positions and lane-group geometry; they do not require SUMO
`laneAreaDetector` elements.

City route generation samples origins and destinations from all normal drivable
edges, weighted by edge storage. This intentionally creates internal city trips
instead of only boundary-to-boundary through traffic, which tends to overload a
few shortest corridors.

## Movement Graph Design

City movement graphs use directed lane groups backed by one or more normal SUMO
edges. Most unambiguous roads remain one edge per `LaneGroup`, but short
controlled approaches may extend upstream across the immediately preceding
signalized node so detector-like observations are not trapped on OSM/SUMO
turn stubs. True tiny controlled turn stubs that share the same upstream
approach are merged into a shared lane group. This intentionally allows some
detector windows to see farther upstream than the current signal, because
otherwise max-pressure and learned policies can be blind to the queue that is
actually feeding the junction.

One `num_hops` macro-hop means one junction transition:

```text
signalized:    LaneGroup -> Movement -> LaneGroup
unsignalized:  LaneGroup -> LaneGroup
```

Direct `LaneGroup -> LaneGroup` connector edges are created only from legal SUMO
connections across non-controllable/pass-through junctions. Controllable
signalized junctions must be traversed through `Movement` nodes. Traffic lights
with zero or one selectable phase are excluded from the policy control set and
are treated as pass-through topology when connector edges are built.

Connector metadata includes source/target lane groups, via junction, distance
context, freeflow time, bottleneck lane count, and connector type. The first
model pass uses deterministic decay `exp(-freeflow_time_s / 30)` for direct
unsignalized lane-to-lane messages. Signalized lane/movement edges currently use
deterministic unit message weight; their metadata is stored for future decay
variants.

## Inspect Movement Extraction

Run this before training or long evaluations:

```powershell
python scripts\inspect_movement_city.py `
  --cfg configs\munich_small\munich_small.sumocfg
```

The report prints:

```text
traffic lights with selectable phases
lane groups
movements
lane-lane connector edges
pass-through/single-phase traffic lights
per-traffic-light selectable phase counts
unsupported/skipped traffic lights and reasons
suspicious lane groups or movements
connector edges that incorrectly cross controllable signalized junctions
```

Skipped traffic lights should be inspected before training. A small number can
be acceptable for early visual checks if they are not intended control nodes.

## Visualize The Movement Graph

```powershell
python scripts\visualize_movement_graph.py `
  --cfg configs\munich_small\munich_small.sumocfg `
  --out reports\munich_small_movement_graph.html `
  --open
```

Use the SUMO map layout first to check whether LaneGroups, detector windows, and
green unsignalized connector edges follow the SUMO road topology. Use the
relaxed layout for dense city centers where labels overlap. Clicking a LaneGroup
shows incoming/outgoing connector metadata, including via junction, freeflow
time, and distance context.

After architecture changes, old learned checkpoints and saved IL/PPO datasets
may be incompatible or may not generalize. Regenerate datasets and checkpoints
for the current graph schema before training or evaluating learned policies.

## Run Baselines Visually

```powershell
python scripts\run.py `
  --cfg configs\munich_small\munich_small.sumocfg `
  --method max-pressure `
  --gui `
  --demand-scale 4.5 `
  --time-to-teleport -1
```

Queue baseline:

```powershell
python scripts\run.py `
  --cfg configs\munich_small\munich_small.sumocfg `
  --method queue `
  --gui
```

## Evaluate Baselines

```powershell
python scripts\eval_policy.py `
  --cfg configs\munich_small\munich_small.sumocfg `
  --policies max-pressure queue `
  --seeds 100 101 102 `
  --steps 1200 `
  --demand-scale 4.5 `
  --time-to-teleport -1
```

For the first Mannheim-style city checks, demand scale around `4.5` produced
plausible visible flow after detector fixes. Use `3.5..5.0` as the initial
multi-city training range, then inspect SUMO-GUI behavior before city PPO
training. Do not keep increasing demand just to compensate for one degenerate
approach; fix or exclude that topology issue first.
