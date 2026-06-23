# City / OSM Movement Network Usage

The OSM build path now targets the movement-based controller, not the legacy
fixed-intersection pipeline. It writes movement-safe traffic-light programs,
city O-D flows, an empty additional file, and a SUMO config.

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

Using an existing OSM file:

```powershell
python scripts\build_network.py `
  --osm configs\munich_small\munich_small.osm `
  --out-dir configs\munich_small `
  --name munich_small
```

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

City movement graphs no longer depend on broad unsignalized corridor
contraction. With a SUMO network file, each `LaneGroup` is a directed normal
SUMO edge. Controlled multi-phase traffic lights create `Movement` nodes, and
movement scores remain the learned policy output.

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

Use the SUMO map layout first to check whether one-edge LaneGroups and green
unsignalized connector edges follow the SUMO road topology. Use the relaxed
layout for dense city centers where labels overlap. Clicking a LaneGroup shows
incoming/outgoing connector metadata, including via junction, freeflow time, and
distance context.

After architecture changes, old learned checkpoints and saved IL/PPO datasets
may be incompatible or may not generalize. Regenerate datasets and checkpoints
for the current graph schema before training or evaluating learned policies.

## Run Baselines Visually

```powershell
python scripts\run.py `
  --cfg configs\munich_small\munich_small.sumocfg `
  --method max-pressure `
  --gui `
  --demand-scale 0.65 `
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
  --demand-scale 0.65 `
  --time-to-teleport -1
```

For first city checks, keep demand simple. Try fixed values around `0.4`,
`0.65`, and `0.85`, then inspect SUMO-GUI behavior before city PPO training.
