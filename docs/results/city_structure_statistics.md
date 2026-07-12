# Structural statistics for the five city scenarios

| City | Role | Junctions | Signalized nodes | Policy controllers | LaneGroups | Movements | Unsignalized connectors | Selectable phases | Phases/controller (mean; median; range) | Typed message edges | Lane length (km) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Karlsruhe | PPO rollout | 141 | 42 | 41 | 286 | 271 | 319 | 136 | 3.32; 3.0; 2-10 | 1084 | 100.39 |
| Mannheim | PPO rollout | 417 | 100 | 84 | 708 | 467 | 993 | 254 | 3.02; 2.0; 2-7 | 1868 | 134.08 |
| Stuttgart | PPO rollout | 290 | 71 | 69 | 519 | 508 | 721 | 238 | 3.45; 3.0; 2-12 | 2032 | 131.54 |
| Heidelberg | PPO rollout | 244 | 55 | 56 | 408 | 365 | 460 | 177 | 3.16; 3.0; 2-10 | 1460 | 86.84 |
| Freiburg | held out from PPO rollouts | 231 | 63 | 58 | 425 | 416 | 585 | 205 | 3.53; 3.0; 2-11 | 1664 | 109.47 |

## Definitions and reproducibility

- `Junctions` counts non-internal `<junction>` elements in the saved SUMO network.
- `Signalized` counts those junctions whose SUMO type begins with `traffic_light`.
- `Policy controllers` counts extracted traffic-light programs with more than one selectable phase; these are the independent policy action sites.
- Signalized-node and controller counts can differ because SUMO controller structures may represent clustered or joined junctions.
- LaneGroups, Movements, connectors, and phase incidences are rebuilt through the current runtime graph extraction.
- Each Movement contributes one edge to each of the four typed LaneGroup/Movement relations.
- `Lane length` sums the lengths of all lanes on non-internal SUMO edges; it is lane-kilometres, not centreline road length.
- Full-precision values and controller/pass-through/unsupported counts are in `city_structure_statistics.csv`.

Generated with:

```powershell
uv run python scripts\analyze_city_structure.py
```
