from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import (  # noqa: E402
    MovementDatasetSample,
    MovementEdgeIndices,
    StoredPhaseIncidence,
    save_jsonl_samples,
)
from src.movement.training.il.indexed_jsonl import (  # noqa: E402
    IndexedJsonlDataset,
    train_movement_il_from_indexed_jsonl,
)
from src.movement.training.il.types import MovementILTrainingConfig  # noqa: E402


def test_indexed_jsonl_counts_and_splits_without_materialized_samples(tmp_path: Path) -> None:
    dataset_path = tmp_path / 'samples.jsonl'
    index_cache_path = tmp_path / 'cache' / 'jsonl_index.pt'
    save_jsonl_samples(dataset_path, _samples())

    dataset = IndexedJsonlDataset(dataset_path, index_cache_path=index_cache_path)
    cached_dataset = IndexedJsonlDataset(dataset_path, index_cache_path=index_cache_path)
    split = dataset.split_train_validation(validation_fraction=0.5, seed=3, max_train_samples=None)
    stats = dataset.stats(split)

    assert index_cache_path.exists()
    assert stats.file_size_bytes == dataset_path.stat().st_size
    assert stats.sample_count == 4
    assert len(cached_dataset.records) == 4
    assert stats.train_count == 2
    assert stats.validation_count == 2
    assert {(group.city_name, group.collection_seed) for group in stats.groups} == {
        ('karlsruhe_oststadt', 1),
        ('karlsruhe_oststadt', 2),
        ('mannheim_innenstadt', 1),
        ('mannheim_innenstadt', 2),
    }


def test_indexed_jsonl_training_logs_progress_and_checkpoints(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / 'samples.jsonl'
    save_jsonl_samples(dataset_path, _samples())
    checkpoint_dir = tmp_path / 'ckpt'
    config = MovementILTrainingConfig(
        epochs=1,
        lr=0.01,
        hidden_dim=8,
        checkpoint_dir=checkpoint_dir,
        seed=1,
        num_hops=0,
        samples_per_batch=2,
        progress_every_batches=1,
        progress_every_seconds=0,
        checkpoint_every_epochs=1,
    )

    train_movement_il_from_indexed_jsonl(
        dataset_path=dataset_path,
        config=config,
        observer=None,
        validation_fraction=0.5,
        max_train_samples=None,
    )

    output = capsys.readouterr().out
    assert 'dataset path=' in output
    assert 'train=2 validation=2' in output
    assert 'epoch_start=1/1' in output
    assert 'batch=1/1' in output
    assert 'epoch_end=1/1' in output
    assert (checkpoint_dir / 'movement_policy_last.pt').exists()
    assert (checkpoint_dir / 'movement_policy_best.pt').exists()
    assert (checkpoint_dir / 'movement_tensor_cache' / 'raw_samples' / 'sample_00000000.pt').exists()
    assert (checkpoint_dir / 'movement_tensor_cache' / 'raw_samples' / 'preparation_state.pt').exists()
    assert (checkpoint_dir / 'movement_tensor_cache' / 'jsonl_index.pt').exists()


def _samples() -> tuple[MovementDatasetSample, ...]:
    base_sample = _sample()
    return (
        replace(base_sample, metadata={'city_name': 'karlsruhe_oststadt', 'collection_seed': 1}),
        replace(base_sample, metadata={'city_name': 'karlsruhe_oststadt', 'collection_seed': 2}),
        replace(base_sample, metadata={'city_name': 'mannheim_innenstadt', 'collection_seed': 1}),
        replace(base_sample, metadata={'city_name': 'mannheim_innenstadt', 'collection_seed': 2}),
    )


def _sample() -> MovementDatasetSample:
    return MovementDatasetSample(
        x_lane=(
            (10.0, 10.0),
            (20.0, 20.0),
            (30.0, 30.0),
        ),
        x_movement=(
            (1.0, 2.0, 3.0, 0.0),
            (4.0, 5.0, 6.0, 1.0),
        ),
        edge_indices=MovementEdgeIndices(
            input_lane_to_movement=((0, 0), (1, 1)),
            output_lane_to_movement=((1, 0), (2, 1)),
            movement_to_input_lane=((0, 0), (1, 1)),
            movement_to_output_lane=((0, 1), (1, 2)),
        ),
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0, 1),
                movement_ids=(0, 1),
                rows=((1, 0), (0, 1)),
            )
        },
        teacher_movement_scores=(7.0, -3.0),
        teacher_selected_phase_by_tls={'J0': 0},
        metadata={'city_name': 'karlsruhe_oststadt', 'collection_seed': 1},
    )
