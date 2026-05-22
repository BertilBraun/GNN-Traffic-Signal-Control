from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.gat_policy import GATPolicy
from src.training.imitation import load_checkpoint


def test_load_checkpoint_accepts_rl_latest_checkpoint(tmp_path: Path) -> None:
    torch.save(GATPolicy().state_dict(), tmp_path / "rl_policy_latest.pt")
    np.savez(
        tmp_path / "normalizer.npz",
        n=np.array(3),
        mean=np.zeros(41, dtype=np.float32),
        M2=np.ones(41, dtype=np.float32),
    )

    model, norm_state = load_checkpoint(str(tmp_path), device="cpu")

    assert isinstance(model, GATPolicy)
    assert norm_state["n"] == 3
