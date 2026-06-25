from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import MovementDatasetSample, MovementEdgeIndices, StoredPhaseIncidence  # noqa: E402
from src.movement.training.il.batching import (  # noqa: E402
    CityBalancedBatchPlanner,
    RandomBatchPlanner,
    split_train_validation_by_city_seed,
)


def test_city_balanced_batch_planner_oversamples_smaller_cities_evenly() -> None:
    samples = (
        _sample('karlsruhe_oststadt'),
        _sample('karlsruhe_oststadt'),
        _sample('karlsruhe_oststadt'),
        _sample('mannheim_innenstadt'),
    )
    planner = CityBalancedBatchPlanner(samples_per_batch=2, seed=123)

    batches = planner.epoch_batches(samples=samples, epoch=0)
    planned_city_names = tuple(
        samples[sample_index].metadata['city_name'] for batch in batches for sample_index in batch
    )

    assert len(planned_city_names) == 6
    assert planned_city_names.count('karlsruhe_oststadt') == 3
    assert planned_city_names.count('mannheim_innenstadt') == 3
    assert all(len(batch) <= 2 for batch in batches)


def test_city_balanced_batch_planner_requires_city_metadata() -> None:
    planner = CityBalancedBatchPlanner(samples_per_batch=2, seed=123)

    with pytest.raises(ValueError, match='city_name'):
        planner.epoch_batches(samples=(_sample(None),), epoch=0)


def test_random_batch_planner_visits_each_sample_once() -> None:
    samples = tuple(_sample('karlsruhe_oststadt') for _sample_index in range(5))
    planner = RandomBatchPlanner(samples_per_batch=2, seed=123)

    batches = planner.epoch_batches(samples=samples, epoch=0)
    sample_indices = tuple(sample_index for batch in batches for sample_index in batch)

    assert sorted(sample_indices) == [0, 1, 2, 3, 4]
    assert all(len(batch) <= 2 for batch in batches)


def test_split_train_validation_by_city_seed_holds_out_seed_groups_per_city() -> None:
    samples = (
        _sample('karlsruhe_oststadt', collection_seed=1),
        _sample('karlsruhe_oststadt', collection_seed=1),
        _sample('karlsruhe_oststadt', collection_seed=2),
        _sample('mannheim_innenstadt', collection_seed=3),
        _sample('mannheim_innenstadt', collection_seed=4),
    )

    split_samples = split_train_validation_by_city_seed(
        samples=samples,
        validation_fraction=0.5,
        seed=42,
    )

    assert split_samples.training_samples
    assert split_samples.validation_samples
    assert {sample.metadata['city_name'] for sample in split_samples.validation_samples} == {
        'karlsruhe_oststadt',
        'mannheim_innenstadt',
    }
    validation_keys = {
        (sample.metadata['city_name'], sample.metadata['collection_seed'])
        for sample in split_samples.validation_samples
    }
    training_keys = {
        (sample.metadata['city_name'], sample.metadata['collection_seed']) for sample in split_samples.training_samples
    }
    assert validation_keys.isdisjoint(training_keys)


def _sample(city_name: str | None, collection_seed: int = 0) -> MovementDatasetSample:
    metadata = {} if city_name is None else {'city_name': city_name, 'collection_seed': collection_seed}
    return MovementDatasetSample(
        x_lane=((1.0, 2.0),),
        x_movement=((3.0, 4.0),),
        edge_indices=MovementEdgeIndices(
            input_lane_to_movement=((0, 0),),
            output_lane_to_movement=((0, 0),),
            movement_to_input_lane=((0, 0),),
            movement_to_output_lane=((0, 0),),
        ),
        phase_incidences={
            'J0': StoredPhaseIncidence(
                sumo_phase_indices=(0,),
                movement_ids=(0,),
                rows=((1,),),
            )
        },
        teacher_movement_scores=(1.0,),
        teacher_selected_phase_by_tls={'J0': 0},
        metadata=metadata,
    )
