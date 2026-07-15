from pathlib import Path
import random
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.evaluation.runner import (  # noqa: E402
    _fixed_time_states,
    _queue_pressure_states,
    _sticky_best_phase_index,
    _uniform_random_phase_state,
)
from src.movement.policies import MovementScoringMethod  # noqa: E402
from src.movement.schema import (  # noqa: E402
    PhaseState,
    SelectablePhase,
    SumoPhaseIndex,
    TrafficLightId,
    TrafficLightProgram,
)


def test_sticky_best_phase_retains_current_phase_on_tie() -> None:
    assert _sticky_best_phase_index(phase_scores=(4.0, 4.0, 2.0), current_local_index=1) == 1


def test_sticky_best_phase_selects_strictly_better_phase() -> None:
    assert _sticky_best_phase_index(phase_scores=(4.0, 5.0, 2.0), current_local_index=0) == 1


def test_uniform_random_phase_is_seeded_and_respects_allowed_states() -> None:
    program = _program()
    first_generator = random.Random(100)
    second_generator = random.Random(100)
    allowed_states = frozenset({'rGr', 'rrG'})

    first_sequence = tuple(
        _uniform_random_phase_state(program, allowed_states, first_generator) for _index in range(20)
    )
    second_sequence = tuple(
        _uniform_random_phase_state(program, allowed_states, second_generator) for _index in range(20)
    )

    assert first_sequence == second_sequence
    assert set(first_sequence) == allowed_states


def test_fixed_time_phase_cycles_every_two_decisions() -> None:
    program = _program()
    programs = {'J0': program}
    allowed_states = {'J0': frozenset(phase.state for phase in program.selectable_phases)}

    states = tuple(
        _fixed_time_states(
            programs=programs,
            accepted_targets={'J0': 'Grr'},
            allowed_target_states_by_traffic_light=allowed_states,
            decision_index=decision_index,
            fixed_time_phase_decisions=2,
        )['J0']
        for decision_index in range(7)
    )

    assert states == ('Grr', 'Grr', 'rGr', 'rGr', 'rrG', 'rrG', 'Grr')


def test_queue_pressure_phase_is_retained_between_scoring_decisions() -> None:
    program = _program()

    states = _queue_pressure_states(
        programs={'J0': program},
        baseline_context=None,
        accepted_targets={'J0': 'rGr'},
        scoring_method=MovementScoringMethod.QUEUE,
        decision_index=1,
        phase_decisions=2,
    )

    assert states == {'J0': 'rGr'}


def _program() -> TrafficLightProgram:
    return TrafficLightProgram(
        traffic_light_id=TrafficLightId('J0'),
        movements=(),
        selectable_phases=tuple(
            SelectablePhase(
                sumo_phase_index=SumoPhaseIndex(index),
                state=PhaseState(state),
                enabled_movement_indices=(),
            )
            for index, state in enumerate(('Grr', 'rGr', 'rrG'))
        ),
    )
