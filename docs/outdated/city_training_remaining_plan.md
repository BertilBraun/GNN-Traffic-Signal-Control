# City Multi-Network Training Remaining Plan

This plan covers the remaining work needed to move from cleaned city networks
to long-running multi-city imitation learning and PPO fine-tuning.

The implementation should be done one step at a time. Each step has its own
acceptance checks and should be committed separately once it passes. Generated
networks, prune files, SUMO outputs, reports, datasets, TensorBoard runs, and
checkpoints remain local artifacts unless explicitly requested otherwise.

## Current Starting Point

The city network generation path has been reworked:

* each city has a build YAML under `configs/<city>/<city>.build.yaml`;
* prune recipes can be regenerated through the network workbench;
* current city demand is expected to be valid around demand scale `0.8..1.2`;
* `scripts/run.py` uses graph-level max-pressure and queue baselines;
* `scripts/collect_il_data.py` collects single-city max-pressure imitation
  samples;
* `scripts/train_il.py` trains from a single collected dataset and can evaluate
  on one config;
* `scripts/train_rl.py` and `src/movement/training/ppo/` support parallel PPO
  rollouts, but are still centered on one SUMO config.

The remaining work is orchestration and reproducibility rather than topology
construction.

## Coding Standards To Keep In Scope

Follow `docs/CODING_STANDARDS.md` while implementing this plan:

* all functions must have full parameter and return type annotations;
* use dataclasses for internal structured data and Pydantic models for
  serialized YAML/JSON config data;
* do not pass structured state as raw dictionaries with string keys;
* use enums for fixed value sets such as split names, policy names, and demand
  sampling modes;
* avoid abbreviations in new code: prefer `configuration_path` over `cfg_path`
  in newly introduced APIs;
* avoid silent defaults in new internal functions; require explicit values at
  call sites;
* keep orchestration modules thin and split parsing, scheduling, collection,
  evaluation, logging, and persistence into focused helpers;
* do not add test fakes or smoke-only branches to production code;
* run `ruff format`, `ruff check --fix`, and focused tests before every commit.

Existing code may still contain older names such as `cfg_path`; new code should
not spread those conventions further unless it is only adapting to an existing
public function signature.

## Target Training Shape

Initial city training split:

```text
train:
  karlsruhe_oststadt
  mannheim_innenstadt
  stuttgart_mitte
  heidelberg_bergheim

held-out evaluation:
  freiburg_altstadt
```

Initial rollout parallelism:

```text
8 workers total
2 workers per training city
```

Demand:

```text
training demand scale: sampled in [0.8, 1.2]
evaluation demand scales: fixed values such as 0.8, 1.0, 1.2
time_to_teleport: -1
```

Long-run target:

```text
IL warm start first
PPO fine-tuning from IL checkpoint
periodic train-city and held-out evaluation
about 1000 PPO iterations if metrics keep improving
```

## Step 1: Add Experiment Configuration

Add a committed experiment YAML, for example:

```text
configs/training/city_first_pass.yaml
```

The YAML should reference generated city configs and build YAMLs, not duplicate
all build details.

Suggested shape:

```yaml
name: city_first_pass

cities:
  - name: karlsruhe_oststadt
    split: train
    sumo_config: configs/karlsruhe_oststadt/karlsruhe_oststadt.sumocfg
    build_config: configs/karlsruhe_oststadt/karlsruhe_oststadt.build.yaml
    rollout_workers: 2
  - name: mannheim_innenstadt
    split: train
    sumo_config: configs/mannheim_innenstadt/mannheim_innenstadt.sumocfg
    build_config: configs/mannheim_innenstadt/mannheim_innenstadt.build.yaml
    rollout_workers: 2
  - name: stuttgart_mitte
    split: train
    sumo_config: configs/stuttgart_mitte/stuttgart_mitte.sumocfg
    build_config: configs/stuttgart_mitte/stuttgart_mitte.build.yaml
    rollout_workers: 2
  - name: heidelberg_bergheim
    split: train
    sumo_config: configs/heidelberg_bergheim/heidelberg_bergheim.sumocfg
    build_config: configs/heidelberg_bergheim/heidelberg_bergheim.build.yaml
    rollout_workers: 2
  - name: freiburg_altstadt
    split: held_out
    sumo_config: configs/freiburg_altstadt/freiburg_altstadt.sumocfg
    build_config: configs/freiburg_altstadt/freiburg_altstadt.build.yaml
    rollout_workers: 0

simulation:
  decision_interval: 10
  time_to_teleport: -1
  yellow_duration: 3
  min_green_steps: 2
  initial_occupancy_min: 0.05
  initial_occupancy_max: 0.08

demand:
  train_scale_min: 0.8
  train_scale_max: 1.2
  eval_scales: [0.8, 1.0, 1.2]

imitation_learning:
  samples_per_city: 4800
  samples_per_simulation: 240
  collection_workers: 8
  epochs: 400
  samples_per_batch: 32
  phase_loss_coefficient: 1.0

ppo:
  iterations: 1000
  steps_per_rollout: 1800
  rollouts_per_update: 8
  rollout_workers: 8
  eval_every_iterations: 25
  save_every_iterations: 25

evaluation:
  policies: [learned, max-pressure, queue]
  seeds: [100, 101, 102]
  steps: 1800
```

Implementation notes:

* create typed Pydantic models for this config;
* reject duplicate city names;
* require exactly one held-out city for the first pass;
* require positive worker counts for training cities;
* require held-out cities to have zero rollout workers;
* validate that referenced files exist, but do not regenerate city configs here.

Acceptance:

* the config loads through a typed parser;
* validation failures are clear `ValueError`s at the config boundary;
* a tiny unit test covers a valid config and at least two invalid configs;
* no training code has changed yet.

## Step 2: Add Multi-City Evaluation First

Build the evaluation safety net before changing training.

Add:

```text
scripts/eval_multi_city.py
src/movement/evaluation/multi_city.py
```

Responsibilities:

* load the experiment config;
* run evaluation on all train cities and held-out cities;
* run `learned`, `max-pressure`, and `queue` policies when requested;
* evaluate each requested seed and demand scale;
* write per-run CSV;
* write aggregate JSON;
* print a compact per-city table;
* optionally write TensorBoard scalars if a log directory is supplied.

Desired command:

```powershell
python scripts\eval_multi_city.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --checkpoint checkpoints\...\movement_policy.pt `
  --output-dir reports\city_first_pass_eval `
  --device cuda
```

Smoke command:

```powershell
python scripts\eval_multi_city.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --policies max-pressure queue `
  --steps 300 `
  --output-dir reports\city_first_pass_eval_smoke
```

Acceptance:

* baselines run without a checkpoint;
* learned policy requires a checkpoint only when `learned` is requested;
* output contains city name, split, policy, seed, demand scale, and metrics;
* Freiburg appears in evaluation and never in training outputs;
* smoke test runs all cities for a short episode.

## Step 3: Add Multi-City IL Collection

Add a multi-city collection entry point instead of overloading the existing
single-city script too heavily.

Suggested files:

```text
scripts/collect_multi_city_il.py
src/movement/training/il/multi_city_collection.py
```

Responsibilities:

* load the experiment config;
* collect only from train cities;
* schedule collection jobs across cities in a balanced way;
* sample demand scales in `[0.8, 1.2]`;
* sample initial occupancy using the existing initial-traffic helper;
* run multiple SUMO processes in parallel;
* write one JSONL file per city and one combined JSONL file;
* include city metadata in every sample.

Desired command:

```powershell
python scripts\collect_multi_city_il.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --output-dir data\il\city_first_pass `
  --workers 8
```

Smoke command:

```powershell
python scripts\collect_multi_city_il.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --output-dir data\il\city_first_pass_smoke `
  --workers 2 `
  --samples-per-city 20
```

Implementation notes:

* use typed job/result dataclasses;
* do not return raw dictionaries from worker processes;
* make failures include city name, seed, and demand scale;
* keep sample metadata serialized as part of the existing dataset schema;
* do not collect held-out Freiburg samples.

Acceptance:

* smoke collection produces samples for all four train cities;
* sample metadata includes city name, split, seed, demand scale, and occupancy;
* combined sample count equals the sum of per-city counts;
* held-out city sample count is zero;
* collection can run with one worker and multiple workers.

## Step 4: Make IL Training City-Balanced

Current IL training consumes one dataset without city-aware balancing. For
multi-city training, dense cities can dominate unless the sampler is explicit.

Responsibilities:

* load multi-city JSONL samples;
* split train/validation samples by city and seed;
* draw batches with balanced city representation;
* log global and per-city validation losses;
* keep the existing model/checkpoint format compatible unless a clean break is
  required.

Desired command:

```powershell
python scripts\train_il.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --data data\il\city_first_pass\combined.jsonl `
  --device cuda `
  --ckpt-dir checkpoints\il\city_first_pass `
  --log-dir runs\il\city_first_pass
```

Implementation notes:

* prefer adding a new typed training configuration path over many new CLI flags;
* city balancing should be deterministic under the configured seed;
* TensorBoard tags should include city names for validation metrics;
* periodic evaluation should call the multi-city evaluator from Step 2.

Acceptance:

* a tiny synthetic dataset test proves balanced city sampling;
* a two-epoch smoke train works on the collected smoke dataset;
* checkpoint loads in `scripts/run.py --method learned`;
* periodic evaluation can run train cities plus Freiburg.

## Step 5: Adapt PPO To Multi-City Rollouts

The PPO implementation already has parallel worker processes, but each rollout
currently uses one config path. Extend the rollout request to include city
identity and SUMO config path.

Responsibilities:

* represent rollout city assignment explicitly;
* schedule two workers per train city for the first run;
* aggregate all city rollouts into one PPO update;
* preserve city metadata through rollout metrics;
* log per-city rollout reward, throughput, completion, teleport count, and
  wait density;
* keep Freiburg eval-only.

Desired command:

```powershell
python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --il-checkpoint checkpoints\il\city_first_pass\movement_policy.pt `
  --device cuda `
  --iterations 1000 `
  --num-workers 8 `
  --ckpt-dir checkpoints\rl\city_first_pass `
  --log-dir runs\rl\city_first_pass
```

Smoke command:

```powershell
python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --il-checkpoint checkpoints\il\city_first_pass\movement_policy.pt `
  --device cpu `
  --iterations 2 `
  --num-workers 4 `
  --steps-per-rollout 120
```

Implementation notes:

* do not randomize city choice implicitly in the worker;
* build a deterministic rollout schedule in the parent process;
* keep city-balanced rollout counts per PPO update;
* include city name in rollout worker exceptions;
* skip or retry failed rollouts deliberately, not silently;
* make resume restore the same schedule for the resumed iteration.

Acceptance:

* two-iteration PPO smoke run completes with more than one city;
* TensorBoard has per-city rollout metrics;
* evaluation runs on all train cities plus Freiburg;
* checkpoint resume works for at least one additional iteration;
* held-out Freiburg is not used by rollout collection.

## Step 6: Harden Long-Run Resume And Logging

Before renting a GPU machine, make long runs robust.

Responsibilities:

* checkpoint the experiment config text or config hash;
* store current iteration, optimizer state, random states, and normalizers;
* write latest and numbered checkpoints;
* keep best-by-held-out-evaluation checkpoint separately;
* write machine-readable run metadata;
* make TensorBoard tags stable and readable.

Recommended TensorBoard tag groups:

```text
train/loss
train/policy_loss
train/value_loss
train/entropy
train/approx_kl
rollout/<city>/reward_mean
rollout/<city>/completion_rate
rollout/<city>/teleport_count
eval/<split>/<city>/<policy>/average_wait_density_s_per_m
eval/<split>/<city>/<policy>/throughput_per_hour
eval/aggregate/<split>/<policy>/average_wait_density_s_per_m
```

Acceptance:

* an interrupted PPO smoke run can resume;
* resumed run appends to the same TensorBoard run;
* best held-out checkpoint is updated only by Freiburg learned-policy metrics;
* baseline evaluation records are cached or clearly separated from learned
  policy records.

## Step 7: Run Pre-Rental Validation

Run this before starting a long remote run.

Commands:

```powershell
ruff format scripts src tests
ruff check --fix scripts src tests
python -m pytest -q
```

Then run small integration smokes:

```powershell
python scripts\eval_multi_city.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --policies max-pressure queue `
  --steps 300 `
  --output-dir reports\city_first_pass_eval_smoke

python scripts\collect_multi_city_il.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --output-dir data\il\city_first_pass_smoke `
  --workers 2 `
  --samples-per-city 20

python scripts\train_il.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --data data\il\city_first_pass_smoke\combined.jsonl `
  --epochs 2 `
  --device cpu

python scripts\train_rl.py `
  --experiment-config configs\training\city_first_pass.yaml `
  --il-checkpoint checkpoints\il\city_first_pass_smoke\movement_policy.pt `
  --iterations 2 `
  --num-workers 4 `
  --steps-per-rollout 120 `
  --device cpu
```

Acceptance:

* no SUMO route failures;
* no worker deadlocks;
* all commands produce clear output artifacts;
* TensorBoard logs contain per-city metrics;
* all checkpoints reload;
* generated artifacts remain untracked or ignored.

## Step 8: Start The Long Run

Recommended initial rented machine:

```text
Linux
16-32 CPU cores
32 GB RAM or more
GPU with 16-24 GB VRAM
fast local SSD/NVMe
```

Initial run:

```text
8 rollout workers
2 workers per train city
demand scale sampled in [0.8, 1.2]
evaluation every 25 iterations
checkpoint every 25 iterations
target 1000 PPO iterations
```

Scaling rule:

* stay at 8 workers until TensorBoard and utilization look stable;
* try 12 workers if CPU has headroom and GPU waits for rollout collection;
* try 16 workers only if SUMO/TraCI remains stable and memory is comfortable;
* keep the same eval seeds and demand scales so progress is comparable.

Stop or adjust if:

* train cities improve while Freiburg regresses steadily;
* completion rate collapses at demand scale `1.2`;
* teleport count becomes nonzero with teleport disabled unexpectedly;
* rollouts spend most wall time in failed or skipped episodes;
* GPU utilization remains low because rollout collection dominates completely.

## Commit Strategy

Use small commits:

1. experiment config models and YAML;
2. multi-city evaluation;
3. multi-city IL collection;
4. city-balanced IL training;
5. multi-city PPO rollouts;
6. resume/logging hardening;
7. docs and command recipes.

Do not combine PPO changes with dataset collection changes in one commit unless
the shared typed config model forces a small overlap.
