import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.evaluation.runner import _sample_phase_index  # noqa: E402


def test_sample_phase_index_is_reproducible_for_seed() -> None:
    first_generator = random.Random(123)
    second_generator = random.Random(123)

    first_samples = tuple(
        _sample_phase_index(
            phase_scores=(0.0, 1.0, 2.0),
            temperature=1.0,
            random_generator=first_generator,
        )
        for _sample_index in range(20)
    )
    second_samples = tuple(
        _sample_phase_index(
            phase_scores=(0.0, 1.0, 2.0),
            temperature=1.0,
            random_generator=second_generator,
        )
        for _sample_index in range(20)
    )

    assert first_samples == second_samples
    assert set(first_samples) <= {0, 1, 2}


def test_sample_phase_index_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match='temperature must be positive'):
        _sample_phase_index(
            phase_scores=(0.0, 1.0),
            temperature=0.0,
            random_generator=random.Random(123),
        )


def test_sample_phase_index_respects_candidate_indices() -> None:
    random_generator = random.Random(456)
    samples = tuple(
        _sample_phase_index(
            phase_scores=(0.0, 1.0, 2.0),
            temperature=1.0,
            random_generator=random_generator,
            candidate_indices=(0, 1),
        )
        for _sample_index in range(20)
    )

    assert set(samples) <= {0, 1}
