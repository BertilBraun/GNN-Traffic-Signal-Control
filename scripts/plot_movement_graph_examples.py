"""Render static documentation figures from an interactive movement-graph report."""

from __future__ import annotations

import argparse
import json
from math import cos, pi, sin
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_REPORT = PROJECT_ROOT / 'reports' / 'movement_graph_3x3.html'
DEFAULT_CITY_REPORT = PROJECT_ROOT / 'configs' / 'stuttgart_mitte' / 'reports' / 'movement_graph.html'
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / 'docs' / 'assets'
DEFAULT_CITY_JUNCTION = 'cluster_1461415657_1461415710_1461415712_1461415724_#17more'
GRAPH_DATA_PATTERN = re.compile(r'<script id="graph-data" type="application/json">(.*?)</script>', re.DOTALL)


class Junction(BaseModel):
    model_config = ConfigDict(frozen=True)

    junction_id: str
    x: float
    y: float
    is_signalized: bool
    selectable_phase_count: int


class Road(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_junction_id: str
    to_junction_id: str
    is_lane_group: bool


class LaneGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    lane_group_id: int
    from_junction_id: str
    to_junction_id: str


class Movement(BaseModel):
    model_config = ConfigDict(frozen=True)

    movement_id: int
    traffic_light_id: str
    input_lane_group_id: int
    output_lane_group_id: int


class MovementGraphReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_name: str
    junctions: tuple[Junction, ...]
    roads: tuple[Road, ...]
    lane_groups: tuple[LaneGroup, ...]
    movements: tuple[Movement, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-report', type=Path, default=DEFAULT_GRID_REPORT)
    parser.add_argument('--city-report', type=Path, default=DEFAULT_CITY_REPORT)
    parser.add_argument('--city-junction', default=DEFAULT_CITY_JUNCTION)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def load_report(path: Path) -> MovementGraphReport:
    match = GRAPH_DATA_PATTERN.search(path.read_text(encoding='utf-8'))
    if match is None:
        raise ValueError(f'No graph-data payload found in {path}.')
    return MovementGraphReport.model_validate(json.loads(match.group(1)))


def configure_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 220, 'font.size': 10})


def positions_by_junction(report: MovementGraphReport) -> dict[str, tuple[float, float]]:
    return {junction.junction_id: (junction.x, junction.y) for junction in report.junctions}


def positions_by_lane_group(
    report: MovementGraphReport,
    junction_positions: dict[str, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    return {
        lane_group.lane_group_id: (
            (junction_positions[lane_group.from_junction_id][0] + junction_positions[lane_group.to_junction_id][0]) / 2,
            (junction_positions[lane_group.from_junction_id][1] + junction_positions[lane_group.to_junction_id][1]) / 2,
        )
        for lane_group in report.lane_groups
    }


def positions_by_movement(
    movements: tuple[Movement, ...],
    junction_positions: dict[str, tuple[float, float]],
    radius: float,
) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    traffic_light_ids = tuple(dict.fromkeys(movement.traffic_light_id for movement in movements))
    for traffic_light_id in traffic_light_ids:
        local_movements = tuple(movement for movement in movements if movement.traffic_light_id == traffic_light_id)
        center_x, center_y = junction_positions[traffic_light_id]
        for index, movement in enumerate(local_movements):
            angle = 2 * pi * index / len(local_movements)
            positions[movement.movement_id] = (center_x + radius * cos(angle), center_y + radius * sin(angle))
    return positions


def draw_roads(
    axis: Axes,
    roads: tuple[Road, ...],
    junction_positions: dict[str, tuple[float, float]],
) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    for road in roads:
        pair = tuple(sorted((road.from_junction_id, road.to_junction_id)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        start = junction_positions[road.from_junction_id]
        end = junction_positions[road.to_junction_id]
        axis.plot((start[0], end[0]), (start[1], end[1]), color='#AEB6BB', linewidth=2.2, zorder=0)


def draw_graph(
    axis: Axes,
    report: MovementGraphReport,
    movements: tuple[Movement, ...],
    lane_group_ids: frozenset[int],
    movement_radius: float,
    show_all_junctions: bool,
    context_junction_ids: frozenset[str] | None,
) -> None:
    junction_positions = positions_by_junction(report)
    lane_positions = positions_by_lane_group(report, junction_positions)
    movement_positions = positions_by_movement(movements, junction_positions, movement_radius)
    visible_roads = (
        report.roads
        if context_junction_ids is None
        else tuple(
            road
            for road in report.roads
            if road.from_junction_id in context_junction_ids and road.to_junction_id in context_junction_ids
        )
    )
    draw_roads(axis, visible_roads, junction_positions)

    for movement in movements:
        movement_position = movement_positions[movement.movement_id]
        input_position = lane_positions[movement.input_lane_group_id]
        output_position = lane_positions[movement.output_lane_group_id]
        axis.plot(
            (input_position[0], movement_position[0]),
            (input_position[1], movement_position[1]),
            color='#42A5C6',
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
        axis.plot(
            (movement_position[0], output_position[0]),
            (movement_position[1], output_position[1]),
            color='#D99A37',
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    lane_points = tuple(lane_positions[lane_group_id] for lane_group_id in lane_group_ids)
    axis.scatter(
        tuple(point[0] for point in lane_points),
        tuple(point[1] for point in lane_points),
        s=46,
        color='#728F9C',
        edgecolor='white',
        linewidth=0.7,
        label='LaneGroup node',
        zorder=3,
    )
    movement_points = tuple(movement_positions[movement.movement_id] for movement in movements)
    axis.scatter(
        tuple(point[0] for point in movement_points),
        tuple(point[1] for point in movement_points),
        s=30,
        color='#C95B5B',
        edgecolor='white',
        linewidth=0.6,
        label='Movement node',
        zorder=4,
    )
    context_junctions = tuple(
        junction
        for junction in report.junctions
        if (context_junction_ids is None or junction.junction_id in context_junction_ids)
        and (show_all_junctions or junction.is_signalized)
    )
    unsignalized_junctions = tuple(junction for junction in context_junctions if not junction.is_signalized)
    axis.scatter(
        tuple(junction.x for junction in unsignalized_junctions),
        tuple(junction.y for junction in unsignalized_junctions),
        s=25,
        facecolor='white',
        edgecolor='#879196',
        linewidth=1.1,
        label='SUMO junction (context)',
        zorder=4,
    )
    signalized_junctions = tuple(junction for junction in context_junctions if junction.is_signalized)
    axis.scatter(
        tuple(junction.x for junction in signalized_junctions),
        tuple(junction.y for junction in signalized_junctions),
        s=34,
        facecolor='#172126',
        edgecolor='#3DB57A',
        linewidth=1.5,
        label='Signalized junction anchor',
        zorder=5,
    )
    axis.set_aspect('equal')
    axis.grid(False)
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_grid(report: MovementGraphReport, output_directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 8.3))
    movements = report.movements
    lane_group_ids = frozenset(lane_group.lane_group_id for lane_group in report.lane_groups)
    draw_graph(
        axis,
        report,
        movements,
        lane_group_ids,
        movement_radius=28.0,
        show_all_junctions=True,
        context_junction_ids=None,
    )
    axis.set_title('3x3 road layout as a LaneGroup / Movement graph')
    axis.legend(loc='lower center', ncols=4, frameon=False)
    figure.savefig(output_directory / 'movement-graph-3x3.png', bbox_inches='tight')
    plt.close(figure)


def plot_city_junction(
    report: MovementGraphReport,
    traffic_light_id: str,
    output_directory: Path,
) -> None:
    movements = tuple(movement for movement in report.movements if movement.traffic_light_id == traffic_light_id)
    if not movements:
        raise ValueError(f'No movements found for city junction {traffic_light_id}.')
    lane_group_ids = frozenset(
        lane_group_id
        for movement in movements
        for lane_group_id in (movement.input_lane_group_id, movement.output_lane_group_id)
    )
    junction_positions = positions_by_junction(report)
    lane_positions = positions_by_lane_group(report, junction_positions)
    local_points = tuple(lane_positions[lane_group_id] for lane_group_id in lane_group_ids)
    lane_groups_by_id = {lane_group.lane_group_id: lane_group for lane_group in report.lane_groups}
    context_junction_ids = frozenset(
        junction_id
        for lane_group_id in lane_group_ids
        for junction_id in (
            lane_groups_by_id[lane_group_id].from_junction_id,
            lane_groups_by_id[lane_group_id].to_junction_id,
        )
    ) | frozenset((traffic_light_id,))
    center_x, center_y = junction_positions[traffic_light_id]
    local_distances = tuple(max(abs(point[0] - center_x), abs(point[1] - center_y)) for point in local_points)
    local_span = max(1.0, max(local_distances))
    figure, axis = plt.subplots(figsize=(9.2, 7.6))
    draw_graph(
        axis,
        report,
        movements,
        lane_group_ids,
        movement_radius=local_span * 0.10,
        show_all_junctions=True,
        context_junction_ids=context_junction_ids,
    )
    visible_points = (*local_points, (center_x, center_y))
    minimum_x = min(point[0] for point in visible_points)
    maximum_x = max(point[0] for point in visible_points)
    minimum_y = min(point[1] for point in visible_points)
    maximum_y = max(point[1] for point in visible_points)
    padding = max(maximum_x - minimum_x, maximum_y - minimum_y) * 0.12
    axis.set_xlim(minimum_x - padding, maximum_x + padding)
    axis.set_ylim(minimum_y - padding, maximum_y + padding)
    junction = next(junction for junction in report.junctions if junction.junction_id == traffic_light_id)
    approach_count = len({movement.input_lane_group_id for movement in movements})
    exit_count = len({movement.output_lane_group_id for movement in movements})
    axis.set_title(
        'Irregular Stuttgart junction: '
        f'{approach_count} inputs, {exit_count} outputs, {len(movements)} movements, '
        f'{junction.selectable_phase_count} phases'
    )
    axis.legend(loc='lower center', ncols=3, frameon=False)
    figure.savefig(output_directory / 'movement-graph-irregular-junction.png', bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    grid_report = load_report(arguments.grid_report)
    city_report = load_report(arguments.city_report)
    plot_grid(grid_report, arguments.output_dir)
    plot_city_junction(city_report, arguments.city_junction, arguments.output_dir)
    print(f'Wrote movement-graph figures to {arguments.output_dir}')


if __name__ == '__main__':
    main()
