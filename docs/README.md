# Documentation

The maintained project documentation is intentionally small:

- [Architecture and constraints](architecture.md) defines the movement graph, phase aggregation, legal actions, and cross-city parameter sharing.
- [City pipeline](city_pipeline.md) covers reproducible OSM acquisition, shaping, pruning, demand, verification, and generated SUMO configs.
- [Training and evaluation](training_and_evaluation.md) covers imitation learning, scratch and IL-initialized PPO, value warm-up, rewards, checkpointing, and evaluation.
- [Remote/Linux training operations](remote_training.md) covers the maintained launcher, monitoring, recovery, and artifact retrieval.
- [Iteration-85 results](results/city_first_pass_throughput_scratch_32_worker.md) is the scientific result report.

All superseded plans, proposals, runbooks, status pages, coding notes, and historical experiments are stored under [`docs/outdated/`](outdated/). They are retained for provenance and are not part of the current project narrative.
