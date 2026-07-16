# 3×3 Local-Reward PPO Evidence Bundle

This directory contains the downloaded evidence for the completed iteration-60 run described in the
[result report](../../../docs/results/grid_3x3_local_reward_2hop_validation.md).

Run name:
`grid_3x3_local_reward_2hop_5s_warmup15_demand_06_08_rollouts100x200_30_repeat_01`

Remote Git revision: `775e25ad99df885090c454003e9b08c0794db8dd`

- `configs/training/` contains the exact experiment configuration.
- `runs/rl/<run>/` contains launch scripts, metadata, TensorBoard events, and the complete console log.
- `checkpoints/rl/<run>/eval/` contains every six-seed evaluation from iteration 0 through 60.
- `checkpoints/rl/<run>/` contains selected iteration-55 and iteration-60 policy/PPO checkpoints plus
  the final latest checkpoints.
- `SHA256SUMS.txt` provides integrity hashes for every downloaded evidence file.

The remote transfer archive had SHA-256
`42416c39fbebb20377d6c5c575ac02175dcfde58b398eb5eb3f60f97fd92d8f3`.
