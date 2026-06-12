from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.normalization import RunningNormalizer


def test_running_normalizer_fits_and_transforms_columns() -> None:
    normalizer = RunningNormalizer()
    normalizer.update_rows(((1.0, 10.0), (3.0, 14.0), (5.0, 18.0)))

    transformed = normalizer.transform_row((3.0, 18.0))

    assert transformed == (0.0, 1.224745)


def test_running_normalizer_freeze_blocks_later_updates() -> None:
    normalizer = RunningNormalizer()
    normalizer.update_rows(((1.0,), (3.0,)))
    normalizer.freeze()
    normalizer.update_rows(((100.0,),))

    assert normalizer.count == 2
    assert normalizer.transform_row((2.0,)) == (0.0,)
