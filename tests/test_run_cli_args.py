from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts import run


def test_run_cli_accepts_learned_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--method",
            "learned",
            "--checkpoint",
            "checkpoints/il/example/movement_policy_best.pt",
        ],
    )

    args = run.parse_args()

    assert args.method == "learned"
    assert args.checkpoint == Path("checkpoints/il/example/movement_policy_best.pt")
