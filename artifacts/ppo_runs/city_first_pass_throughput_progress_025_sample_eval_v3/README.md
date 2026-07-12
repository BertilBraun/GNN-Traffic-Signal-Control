# Iteration-85 Evidence Bundle

This directory contains the reproducibility evidence for the selected iteration-85 checkpoint described in the [result report](../../../docs/results/city_first_pass_throughput_scratch_32_worker.md).

- `selected_iteration_0085/` contains the policy-only and complete PPO checkpoints plus the six-seed evaluation exports.
- `tensorboard_through_iter_0085/` contains the curated TensorBoard event stream ending at the selected checkpoint.
- `runs/train_stdout.log` is the downloaded training console cut immediately after the completed iteration-85 evaluation.
- `runs/run_metadata.json` records the original command configuration and output locations.
- `SHA256SUMS.txt` provides integrity hashes for every evidence file.

No post-iteration-85 TensorBoard or console data is included in this presentation bundle.
