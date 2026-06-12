from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.movement.dataset import MovementDatasetSample
from src.movement.training.il import (
    ZeroHopTrainingConfig,
    load_zero_hop_checkpoint,
    normalizer_from_state,
    tensors_from_sample,
    train_zero_hop_il,
)
from src.movement.models.zero_hop import ZeroHopMovementScorer


def _sample() -> MovementDatasetSample:
    return MovementDatasetSample(
        x_lane=(
            (10.0, 10.0),
            (20.0, 20.0),
            (30.0, 30.0),
        ),
        x_movement=(
            (1.0, 2.0, 3.0, 0.0, 1.0),
            (4.0, 5.0, 6.0, 1.0, 2.0),
        ),
        edge_index_dict={
            "input_lane_to_movement": ((0, 0), (1, 1)),
            "output_lane_to_movement": ((1, 0), (2, 1)),
            "movement_to_input_lane": ((0, 0), (1, 1)),
            "movement_to_output_lane": ((0, 1), (1, 2)),
        },
        phase_incidences={
            "J0": {
                "sumo_phase_indices": (0, 1),
                "movement_ids": (0, 1),
                "rows": ((1, 0), (0, 1)),
            }
        },
        teacher_movement_scores=(7.0, -3.0),
        teacher_selected_phase_by_tls={"J0": 0},
        metadata={"network_id": "unit"},
    )


def test_zero_hop_model_scores_one_value_per_movement() -> None:
    model = ZeroHopMovementScorer(
        lane_feature_dim=2,
        movement_feature_dim=5,
        hidden_dim=8,
    )

    scores = model(
        x_lane=torch.tensor(_sample().x_lane, dtype=torch.float32),
        x_movement=torch.tensor(_sample().x_movement, dtype=torch.float32),
    )

    assert tuple(scores.shape) == (2,)


def test_zero_hop_il_overfits_tiny_dataset_and_loads_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    result = train_zero_hop_il(
        samples=[_sample()],
        config=ZeroHopTrainingConfig(
            epochs=250,
            lr=0.03,
            hidden_dim=32,
            checkpoint_dir=checkpoint_dir,
            seed=7,
            num_hops=0,
        ),
    )

    assert result.final_loss < 0.05
    assert (checkpoint_dir / "movement_policy_last.pt").exists()
    assert (checkpoint_dir / "movement_policy_best.pt").exists()
    assert (checkpoint_dir / "zero_hop_il.pt").exists()
    loaded_model, metadata = load_zero_hop_checkpoint(checkpoint_dir / "zero_hop_il.pt")
    loaded_model.eval()
    with torch.no_grad():
        x_lane, x_movement, _target = tensors_from_sample(
            sample=_sample(),
            lane_normalizer=normalizer_from_state(metadata["lane_normalizer"]),
            movement_normalizer=normalizer_from_state(metadata["movement_normalizer"]),
        )
        scores = loaded_model(
            x_lane=x_lane,
            x_movement=x_movement,
        )

    assert torch.allclose(scores, torch.tensor([7.0, -3.0]), atol=0.3)
    assert metadata["lane_feature_dim"] == 2
    assert metadata["movement_feature_dim"] == 5
    assert metadata["num_hops"] == 0


def test_one_hop_il_trains_and_saves_hop_metadata(tmp_path: Path) -> None:
    result = train_zero_hop_il(
        samples=[_sample()],
        config=ZeroHopTrainingConfig(
            epochs=10,
            lr=0.01,
            hidden_dim=16,
            checkpoint_dir=tmp_path / "ckpt",
            seed=3,
            num_hops=1,
        ),
    )

    _model, metadata = load_zero_hop_checkpoint(result.checkpoint_path)

    assert metadata["num_hops"] == 1


def test_zero_hop_il_reports_progress_when_requested(tmp_path: Path, capsys) -> None:
    train_zero_hop_il(
        samples=[_sample()],
        config=ZeroHopTrainingConfig(
            epochs=3,
            lr=0.01,
            hidden_dim=8,
            checkpoint_dir=tmp_path / "ckpt",
            progress_every=2,
            seed=1,
            num_hops=0,
        ),
    )

    output = capsys.readouterr().out
    assert "epoch=2/3" in output
    assert "epoch=3/3" in output
    assert "loss=" in output
