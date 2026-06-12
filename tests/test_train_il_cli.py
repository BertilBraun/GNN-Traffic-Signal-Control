from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import train_il


def test_train_il_cli_accepts_dataset_and_checkpoint_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_il.py",
            "--data",
            "data/il/samples.jsonl",
            "--epochs",
            "3",
            "--progress-every",
            "1",
            "--num-hops",
            "1",
            "--ckpt-dir",
            "checkpoints/il/unit",
        ],
    )

    args = train_il.parse_args()

    assert args.data == Path("data/il/samples.jsonl")
    assert args.epochs == 3
    assert args.progress_every == 1
    assert args.num_hops == 1
    assert args.ckpt_dir == Path("checkpoints/il/unit")


def test_train_il_cli_accepts_collection_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_il.py",
            "--cfg",
            "configs/grid_3x3_dedicated/grid.sumocfg",
            "--samples",
            "12",
            "--decision-interval",
            "15",
            "--seed",
            "7",
        ],
    )

    args = train_il.parse_args()

    assert args.data is None
    assert args.cfg == Path("configs/grid_3x3_dedicated/grid.sumocfg")
    assert args.samples == 12
    assert args.decision_interval == 15
    assert args.seed == 7
