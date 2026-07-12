# Documentation

## Current project narrative

- [City first-pass scratch PPO result](results/city_first_pass_throughput_scratch_32_worker.md) — scientific source of truth for the selected iteration-85 checkpoint and full training trajectory.
- [Movement-GNN architecture](movement_architecture_overview.md) — graph representation, phase scoring, and cross-city parameter sharing.
- [PPO training](ppo_training.md) — objective, legal-action sampling, reward, rollout, and evaluation semantics.
- [City / OSM usage](city_osm_usage.md) — network construction, shaping, pruning, demand, and inspection workflow.

## Operational references

- [Remote city-first-pass runbook](city_first_pass_remote_runbook.md) — operational commands and recovery notes; not a scientific results report.
- [Network build pipeline plan](network_build_pipeline_plan.md) — design history for the city workbench.
- [Dataset schema](dataset_schema.md) and [grid generation](grid_generation.md) — implementation references.

## Historical material

Superseded plans and status snapshots live under [`docs/outdated/`](outdated/). They remain available for provenance but do not describe the current result. The `docs/superpowers/` material is design and implementation history rather than current project status.
