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
            "--ckpt-dir",
            "checkpoints/il/unit",
        ],
    )

    args = train_il.parse_args()

    assert args.data == Path("data/il/samples.jsonl")
    assert args.epochs == 3
    assert args.ckpt_dir == Path("checkpoints/il/unit")
