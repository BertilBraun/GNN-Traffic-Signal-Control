# City / OSM Movement Network Usage

The OSM build path now targets the movement-based controller, not the legacy
fixed-intersection pipeline. It writes movement-safe traffic-light programs,
city O-D flows, detectors, and a SUMO config.

## Build A City Config

```powershell
python scripts\build_network.py `
  --bbox 48.147,11.568,48.155,11.581 `
  --out-dir configs\munich_small `
  --name munich_small `
  --route-count 120 `
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
per-traffic-light selectable phase counts
unsupported/skipped traffic lights and reasons
suspicious lane groups or movements
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

Use the SUMO map layout first to check whether contracted LaneGroups follow
directed corridors between signalized junctions. Use the relaxed layout for
dense city centers where labels overlap.

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
