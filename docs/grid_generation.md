# Grid Generation

Generate a dedicated-lane 6x6 SUMO grid:

```powershell
python scripts\generate_grid_network.py --rows 6 --cols 6 --out configs\grid_6x6_dedicated
```

Run the movement-aware max-pressure controller headless:

```powershell
python scripts\run_movement_max_pressure.py --cfg configs\grid_6x6_dedicated\grid.sumocfg --steps 1800 --decision-interval 15 --yellow-duration 3 --verbose
```

Run it visually in SUMO-GUI:

```powershell
python scripts\run_movement_max_pressure.py --cfg configs\grid_6x6_dedicated\grid.sumocfg --gui --steps 1800 --decision-interval 15 --yellow-duration 3 --verbose
```

For the smaller 3x3 sample:

```powershell
python scripts\generate_grid_network.py --rows 3 --cols 3 --out configs\grid_3x3_dedicated
python scripts\run_movement_max_pressure.py --cfg configs\grid_3x3_dedicated\grid.sumocfg --gui --verbose
```
