# Grid Generation

Generate a dedicated-lane 6x6 SUMO grid:

```powershell
python scripts\generate_grid_network.py --rows 6 --cols 6 --out configs\grid_6x6_dedicated
```

By default, generated grids use `900` total vehicles/hour distributed across all boundary route flows. This keeps larger grids from becoming overloaded just because they have more entry flows.

Generate a lighter or heavier grid by choosing the total demand explicitly:

```powershell
python scripts\generate_grid_network.py --rows 4 --cols 4 --out configs\grid_4x4_dedicated --demand-vehicles-per-hour 700
```

Grid generation synthesizes maximal green phases from SUMO movement foes plus an additional conflict when movements from different incoming lanes enter the same outgoing edge.

Run the movement-aware max-pressure controller headless:

```powershell
python scripts\run.py --cfg configs\grid_6x6_dedicated\grid.sumocfg --steps 1800 --decision-interval 15 --yellow-duration 3 --min-green-steps 2 --verbose
```

Run it visually in SUMO-GUI:

```powershell
python scripts\run.py --cfg configs\grid_6x6_dedicated\grid.sumocfg --gui --steps 1800 --decision-interval 15 --yellow-duration 3 --min-green-steps 2 --verbose
```

Switch to the longest-queue visual heuristic:

```powershell
python scripts\run.py --cfg configs\grid_6x6_dedicated\grid.sumocfg --gui --method queue --steps 1800 --decision-interval 15 --yellow-duration 3 --min-green-steps 2 --verbose
```

With `--decision-interval 15 --min-green-steps 2`, an accepted green target is held for at least two decision intervals before that junction may switch again.

Inspect synthesized movement conflicts for the 3x3 center junction:

```powershell
python scripts\tools\analyze_movement_conflicts.py --net configs\grid_3x3_dedicated\grid.net.xml --tls N1_1 --mode conflict-edge
```

For the smaller 3x3 sample:

```powershell
python scripts\generate_grid_network.py --rows 3 --cols 3 --out configs\grid_3x3_dedicated
python scripts\run.py --cfg configs\grid_3x3_dedicated\grid.sumocfg --gui --verbose
```
