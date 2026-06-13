from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.dataset import MovementDatasetSample
from src.movement.training.rollout import MovementRolloutBuffer, MovementTransition


def _sample() -> MovementDatasetSample:
    return MovementDatasetSample(
        x_lane=((1.0,),),
        x_movement=((0.0, 1.0, 1.0, 0.0, 0.0),),
        edge_index_dict={},
        phase_incidences={
            'J0': {
                'sumo_phase_indices': (0,),
                'movement_ids': (0,),
                'rows': ((1,),),
            }
        },
        teacher_movement_scores=(0.0,),
        teacher_selected_phase_by_tls={},
        metadata={},
    )


def test_rollout_buffer_computes_mc_returns_for_value_warmup() -> None:
    buffer = MovementRolloutBuffer(
        traffic_light_count=1,
        gamma=0.5,
        lam=0.95,
    )
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            rewards=(1.0,),
            values=(0.0,),
            done=False,
        )
    )
    buffer.add(
        MovementTransition(
            sample=_sample(),
            actions=(0,),
            old_log_probs=(0.0,),
            rewards=(2.0,),
            values=(0.0,),
            done=True,
        )
    )

    buffer.compute_returns_and_advantages(use_mc_targets=True)

    assert buffer.returns is not None
    assert tuple(float(value) for value in buffer.returns[:, 0]) == (2.0, 2.0)
