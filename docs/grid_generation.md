# Grid Generation

Generate a dedicated-lane 6x6 SUMO grid:

```powershell
python scripts\generate_grid_network.py --rows 6 --cols 6 --out configs\grid_6x6_dedicated --phase-mode conflict-edge
```

`conflict-edge` synthesizes maximal green phases from SUMO movement foes plus an additional conflict when two movements enter the same outgoing edge. Use `--phase-mode protected` to generate the smaller hand-defined protected phase set.

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
python scripts\generate_grid_network.py --rows 3 --cols 3 --out configs\grid_3x3_dedicated --phase-mode conflict-edge
python scripts\run.py --cfg configs\grid_3x3_dedicated\grid.sumocfg --gui --verbose
```
