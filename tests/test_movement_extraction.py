from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.extraction import (
    extract_traffic_light_program,
    is_selectable_green_state,
)


def test_selectable_green_state_filters_transitions_and_empty_states() -> None:
    assert is_selectable_green_state('Grg')
    assert is_selectable_green_state('rgr')

    assert not is_selectable_green_state('rrr')
    assert not is_selectable_green_state('yyy')
    assert not is_selectable_green_state('ryr')
    assert not is_selectable_green_state('')


def test_extract_program_maps_green_positions_to_controlled_movements() -> None:
    controlled_links = [
        ('north_0', 'south_0', None),
        ('east_0', 'west_0', ':via_0'),
        ('north_1', 'east_0', None),
    ]
    phase_states = [
        'Grr',
        'yrr',
        'rGr',
        'rrr',
        'grG',
    ]

    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=phase_states,
        controlled_links=controlled_links,
    )

    assert program.traffic_light_id == 'J0'
    assert [movement.signal_index for movement in program.movements] == [0, 1, 2]
    assert program.movements[0].incoming_lane_id == 'north_0'
    assert program.movements[0].outgoing_lane_id == 'south_0'
    assert program.movements[1].via_lane_id == ':via_0'

    assert [phase.sumo_phase_index for phase in program.selectable_phases] == [0, 2, 4]
    assert [phase.enabled_movement_indices for phase in program.selectable_phases] == [
        (0,),
        (1,),
        (0, 2),
    ]


def test_extract_program_rejects_phase_state_with_wrong_signal_count() -> None:
    controlled_links = [('in_0', 'out_0', None), ('in_1', 'out_1', None)]

    try:
        extract_traffic_light_program(
            tls_id='J0',
            phase_states=['G'],
            controlled_links=controlled_links,
        )
    except ValueError as exc:
        assert 'state length' in str(exc)
    else:
        raise AssertionError('expected state length mismatch to raise ValueError')


def test_extract_program_expands_multiple_controlled_links_per_signal_index() -> None:
    controlled_links = [
        [
            ('north_0', 'south_0', None),
            ('north_0', 'east_0', ':via_0'),
        ],
        [('east_0', 'west_0', None)],
    ]

    program = extract_traffic_light_program(
        tls_id='J0',
        phase_states=['Gr', 'rG'],
        controlled_links=controlled_links,
    )

    assert [(m.movement_index, m.signal_index) for m in program.movements] == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]
    assert program.selectable_phases[0].enabled_movement_indices == (0, 1)
    assert program.selectable_phases[1].enabled_movement_indices == (2,)
