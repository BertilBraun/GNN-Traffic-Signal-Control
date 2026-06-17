"""Rewrite Torch checkpoints after training module package moves."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.movement.training.il as il_package  # noqa: E402
import src.movement.training.il.checkpoint as il_checkpoint  # noqa: E402
import src.movement.training.il.tensors as il_tensors  # noqa: E402
import src.movement.training.il.types as il_types  # noqa: E402
import src.movement.training.ppo as ppo_package  # noqa: E402
import src.movement.training.ppo.types as ppo_types  # noqa: E402
import src.movement.training.rollout as rollout  # noqa: E402
import src.movement.training.rollout.math as rollout_math  # noqa: E402
import src.movement.training.rollout.types as rollout_types  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Load a Torch checkpoint with old module paths and re-save it with current module paths.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint to migrate')
    parser.add_argument('--output', type=Path, required=False, help='Output path; omit to rewrite in place')
    parser.add_argument('--backup', type=Path, required=False, help='Backup path for in-place rewrites')
    parser.add_argument('--device', default='cpu', help='Torch map_location used while loading')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_legacy_module_aliases()
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    output_path = args.output or args.checkpoint_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.name}.tmp')
    torch.save(checkpoint, temporary_path)
    if args.output is None and args.backup is not None:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_path.replace(args.backup)
    temporary_path.replace(output_path)
    print(f'Migrated checkpoint: {args.checkpoint_path} -> {output_path}')


def install_legacy_module_aliases() -> None:
    install_il_package_attributes()
    install_ppo_package_attributes()
    module_aliases = {
        'src.movement.training.il_checkpoint': il_checkpoint,
        'src.movement.training.il_tensors': il_tensors,
        'src.movement.training.il_types': il_types,
        'src.movement.training.rollout_math': rollout_math,
        'src.movement.training.rollout_types': rollout_types,
        'src.movement.training.rollout': rollout,
    }
    for legacy_name, current_module in module_aliases.items():
        sys.modules[legacy_name] = current_module


def install_il_package_attributes() -> None:
    il_package.MovementCheckpointMetadata = il_checkpoint.MovementCheckpointMetadata
    il_package.MovementCheckpointPayload = il_checkpoint.MovementCheckpointPayload
    il_package.NormalizerState = il_checkpoint.NormalizerState
    il_package.MovementILLoss = il_types.MovementILLoss
    il_package.MovementILTrainingConfig = il_types.MovementILTrainingConfig
    il_package.MovementILTrainingResult = il_types.MovementILTrainingResult
    il_package.MovementILTrainingSnapshot = il_types.MovementILTrainingSnapshot


def install_ppo_package_attributes() -> None:
    ppo_package.CollectedRollout = ppo_types.CollectedRollout
    ppo_package.IntervalRewardResult = ppo_types.IntervalRewardResult
    ppo_package.MovementPpoCheckpoint = ppo_types.MovementPpoCheckpoint
    ppo_package.MovementPpoConfig = ppo_types.MovementPpoConfig
    ppo_package.MovementPpoTrainingResult = ppo_types.MovementPpoTrainingResult
    ppo_package.PolicyContext = ppo_types.PolicyContext
    ppo_package.RolloutContext = ppo_types.RolloutContext
    ppo_package.RolloutStats = ppo_types.RolloutStats
    ppo_package.TrainingDiagnostics = ppo_types.TrainingDiagnostics
    ppo_package.TrainingEvaluationResult = ppo_types.TrainingEvaluationResult


if __name__ == '__main__':
    main()
