"""Render static documentation figures from an interactive movement-graph report."""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MatplotlibPath
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_REPORT = PROJECT_ROOT / 'reports' / 'movement_graph_3x3.html'
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / 'docs' / 'assets'
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
    junction_ids: tuple[str, ...]


class Movement(BaseModel):
    model_config = ConfigDict(frozen=True)

    movement_id: int
    traffic_light_id: str
    input_lane_group_id: int
    output_lane_group_id: int


class LaneConnector(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_lane_group_id: int
    target_lane_group_id: int
    via_junction_id: str


class MovementGraphReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_name: str
    junctions: tuple[Junction, ...]
    roads: tuple[Road, ...]
    lane_groups: tuple[LaneGroup, ...]
    movements: tuple[Movement, ...]
    lane_connectors: tuple[LaneConnector, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-report', type=Path, default=DEFAULT_GRID_REPORT)
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
    positions: dict[int, tuple[float, float]] = {}
    for lane_group in report.lane_groups:
        points = tuple(junction_positions[junction_id] for junction_id in lane_group.junction_ids)
        segment_lengths = tuple(hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(points, points[1:]))
        half_length = sum(segment_lengths) / 2
        travelled = 0.0
        for start, end, segment_length in zip(points, points[1:], segment_lengths):
            if travelled + segment_length >= half_length:
                ratio = (half_length - travelled) / segment_length if segment_length else 0.0
                direction_x = end[0] - start[0]
                direction_y = end[1] - start[1]
                normalizer = max(1.0, segment_length)
                positions[lane_group.lane_group_id] = (
                    start[0] + direction_x * ratio - direction_y / normalizer * 5.0,
                    start[1] + direction_y * ratio + direction_x / normalizer * 5.0,
                )
                break
            travelled += segment_length
        else:
            positions[lane_group.lane_group_id] = points[0]
    return positions


def positions_by_movement(
    movements: tuple[Movement, ...],
    junction_positions: dict[str, tuple[float, float]],
    lane_positions: dict[int, tuple[float, float]],
    radius: float,
) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    traffic_light_ids = tuple(dict.fromkeys(movement.traffic_light_id for movement in movements))
    for traffic_light_id in traffic_light_ids:
        local_movements = tuple(movement for movement in movements if movement.traffic_light_id == traffic_light_id)
        center_x, center_y = junction_positions[traffic_light_id]
        input_lane_group_ids = tuple(dict.fromkeys(movement.input_lane_group_id for movement in local_movements))
        for input_lane_group_id in input_lane_group_ids:
            same_input = tuple(
                movement for movement in local_movements if movement.input_lane_group_id == input_lane_group_id
            )
            input_x, input_y = lane_positions[input_lane_group_id]
            direction_x = input_x - center_x
            direction_y = input_y - center_y
            length = max(1.0, hypot(direction_x, direction_y))
            unit_x = direction_x / length
            unit_y = direction_y / length
            for index, movement in enumerate(same_input):
                tangent_offset = (index - (len(same_input) - 1) / 2) * 8.0
                positions[movement.movement_id] = (
                    center_x + unit_x * radius - unit_y * tangent_offset,
                    center_y + unit_y * radius + unit_x * tangent_offset,
                )
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


def draw_lane_connectors(
    axis: Axes,
    connectors: tuple[LaneConnector, ...],
    lane_positions: dict[int, tuple[float, float]],
    junction_positions: dict[str, tuple[float, float]],
) -> None:
    for connector in connectors:
        source = lane_positions[connector.source_lane_group_id]
        target = lane_positions[connector.target_lane_group_id]
        junction = junction_positions[connector.via_junction_id]
        path = MatplotlibPath(
            (source, junction, target),
            (MatplotlibPath.MOVETO, MatplotlibPath.CURVE3, MatplotlibPath.CURVE3),
        )
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle='-|>',
            mutation_scale=7,
            color='#58A66C',
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
        )
        axis.add_patch(arrow)


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
    movement_positions = positions_by_movement(movements, junction_positions, lane_positions, movement_radius)
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
    visible_connectors = (
        report.lane_connectors
        if context_junction_ids is None
        else tuple(
            connector for connector in report.lane_connectors if connector.via_junction_id in context_junction_ids
        )
    )
    draw_lane_connectors(axis, visible_connectors, lane_positions, junction_positions)

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
    handles, labels = axis.get_legend_handles_labels()
    connector_handle = Line2D((), (), color='#58A66C', linewidth=2.0)
    axis.legend(
        (connector_handle, *handles),
        ('Unsignalized LaneGroup connector', *labels),
        loc='lower center',
        ncols=3,
        frameon=False,
    )
    figure.savefig(output_directory / 'movement-graph-3x3.png', bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    grid_report = load_report(arguments.grid_report)
    plot_grid(grid_report, arguments.output_dir)
    print(f'Wrote movement-graph figures to {arguments.output_dir}')


if __name__ == '__main__':
    main()
