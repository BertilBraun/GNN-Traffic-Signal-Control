"""Validate routing, occupancy, demand, saturation, teleports, and signal coverage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict
import sumolib
from sumolib.net import Net

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale
from src.movement.evaluation import EvaluationPolicy, run_evaluation_episode
from src.movement.evaluation.runner import lane_inputs_from_net
from src.movement.grid_study import (
    GRID_COVERAGE_SCENARIOS,
    MATCHED_GRID_SCENARIOS,
    controllable_node_ids,
)
from src.movement.initial_traffic import EFFECTIVE_VEHICLE_SPACING_M, generate_initial_traffic_population
from src.movement.runtime import MovementControlRuntime
from src.movement.sumo_backend import SumoBackendKind


class ValidationSuite(str, Enum):
    MATCHED = 'matched'
    COVERAGE = 'coverage'
    ALL = 'all'


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    rows: int
    cols: int


@dataclass(frozen=True)
class ValidationRunSpec:
    scenario: ValidationScenario
    configuration_root: Path
    output_directory: Path
    demand_scales: tuple[float, ...]
    simulation_seed: int
    simulation_steps: int
    skip_simulation: bool


class GridValidationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_name: str
    rows: int
    cols: int
    internal_junction_count: int
    eligible_controller_count: int
    controller_count: int
    signal_coverage: float
    edge_count: int
    lane_count: int
    lane_length_m: float
    route_flow_count: int
    valid_route_count: int
    boundary_source_count: int
    boundary_destination_count: int
    base_demand_vehicles_per_hour: float
    target_initial_occupancy: float | None
    requested_initial_vehicles: int | None
    generated_initial_vehicles: int | None
    realized_initial_occupancy: float | None
    post_warmup_vehicle_count: int | None
    warmup_seconds: int | None
    warmup_teleport_count: int | None
    demand_scale: float | None
    policy: str | None
    throughput_per_hour: float | None
    completion_rate: float | None
    completed_trip_waiting_time_s: float | None
    switches_per_junction_per_minute: float | None
    wait_density_s_per_m: float | None
    evaluation_teleport_count: int | None


class GridValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    records: tuple[GridValidationRecord, ...]


def validate_scenario(
    scenario: ValidationScenario,
    configuration_root: Path,
    output_directory: Path,
    demand_scales: tuple[float, ...],
    simulation_seed: int,
    simulation_steps: int,
    skip_simulation: bool,
) -> tuple[GridValidationRecord, ...]:
    scenario_directory = configuration_root / scenario.name
    sumo_configuration_path = scenario_directory / 'grid.sumocfg'
    network_path = scenario_directory / 'grid.net.xml'
    route_path = scenario_directory / 'grid.rou.xml'
    network = sumolib.net.readNet(str(network_path), withConnections=True)
    route_root = ET.parse(route_path).getroot()
    flows = tuple(route_root.findall('.//flow'))
    lane_ids_by_edge, lane_geometries = lane_inputs_from_net(network_path)
    lane_count = sum(len(lane_ids) for lane_ids in lane_ids_by_edge.values())
    lane_length_m = sum(geometry.length_m * geometry.num_lanes for geometry in lane_geometries.values())
    controller_count = sum(node.getType() == 'traffic_light' for node in network.getNodes())
    eligible_controller_count = len(controllable_node_ids(rows=scenario.rows, cols=scenario.cols))
    valid_route_count = sum(_route_is_valid(network=network, flow=flow) for flow in flows)
    source_nodes = frozenset(_edge_nodes(flow.attrib['from'])[0] for flow in flows)
    destination_nodes = frozenset(_edge_nodes(flow.attrib['to'])[1] for flow in flows)
    base_demand = 3600.0 * sum(float(flow.attrib.get('probability', '0.0')) for flow in flows)
    _plot_topology(
        scenario=scenario,
        scenario_directory=scenario_directory,
        output_path=output_directory / 'topologies' / f'{scenario.name}.png',
    )

    if skip_simulation:
        return (
            _static_record(
                scenario=scenario,
                controller_count=controller_count,
                edge_count=len(lane_ids_by_edge),
                lane_count=lane_count,
                lane_length_m=lane_length_m,
                route_flow_count=len(flows),
                valid_route_count=valid_route_count,
                boundary_source_count=len(source_nodes),
                boundary_destination_count=len(destination_nodes),
                base_demand=base_demand,
            ),
        )

    occupancy_probe = _probe_initial_occupancy(
        sumo_configuration_path=sumo_configuration_path,
        network_path=network_path,
        lane_length_m=lane_length_m,
        seed=simulation_seed,
    )
    records: list[GridValidationRecord] = []
    for demand_scale in demand_scales:
        metrics = run_evaluation_episode(
            cfg_path=sumo_configuration_path,
            policy=EvaluationPolicy.MAX_PRESSURE,
            seed=simulation_seed,
            steps=simulation_steps,
            decision_interval=5,
            yellow_duration=3,
            yellow_start_delay=0,
            min_green_steps=1,
            learned_policy_config=None,
            demand_scale=demand_scale,
            initial_occupancy_min=0.07,
            initial_occupancy_max=0.07,
            warmup_steps=15,
            time_to_teleport=-1,
            fixed_time_phase_duration=10,
            queue_pressure_phase_duration=10,
            backend_kind=SumoBackendKind.TRACI,
        )
        records.append(
            GridValidationRecord(
                scenario_name=scenario.name,
                rows=scenario.rows,
                cols=scenario.cols,
                internal_junction_count=scenario.rows * scenario.cols,
                eligible_controller_count=eligible_controller_count,
                controller_count=controller_count,
                signal_coverage=controller_count / eligible_controller_count,
                edge_count=len(lane_ids_by_edge),
                lane_count=lane_count,
                lane_length_m=lane_length_m,
                route_flow_count=len(flows),
                valid_route_count=valid_route_count,
                boundary_source_count=len(source_nodes),
                boundary_destination_count=len(destination_nodes),
                base_demand_vehicles_per_hour=base_demand,
                target_initial_occupancy=occupancy_probe.target_occupancy,
                requested_initial_vehicles=occupancy_probe.requested_vehicle_count,
                generated_initial_vehicles=occupancy_probe.generated_vehicle_count,
                realized_initial_occupancy=occupancy_probe.realized_initial_occupancy,
                post_warmup_vehicle_count=occupancy_probe.post_warmup_vehicle_count,
                warmup_seconds=15,
                warmup_teleport_count=occupancy_probe.teleport_count,
                demand_scale=demand_scale,
                policy=EvaluationPolicy.MAX_PRESSURE.value,
                throughput_per_hour=metrics.throughput_per_hour,
                completion_rate=metrics.completion_rate,
                completed_trip_waiting_time_s=metrics.average_waiting_time_s,
                switches_per_junction_per_minute=metrics.phase_switch_frequency_per_junction_per_minute,
                wait_density_s_per_m=metrics.average_wait_density_s_per_m,
                evaluation_teleport_count=metrics.teleport_count,
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class InitialOccupancyProbe:
    target_occupancy: float
    requested_vehicle_count: int
    generated_vehicle_count: int
    realized_initial_occupancy: float
    post_warmup_vehicle_count: int
    teleport_count: int


def _probe_initial_occupancy(
    sumo_configuration_path: Path,
    network_path: Path,
    lane_length_m: float,
    seed: int,
) -> InitialOccupancyProbe:
    target_occupancy = 0.07
    initial_population = generate_initial_traffic_population(
        cfg_path=sumo_configuration_path,
        net_path=network_path,
        target_occupancy=target_occupancy,
        seed=seed,
    )
    demand_route_files = route_files_for_demand_scale(
        cfg_path=sumo_configuration_path,
        demand_scale=0.7,
    )
    runtime = MovementControlRuntime(
        cfg_path=sumo_configuration_path,
        gui=False,
        seed=seed,
        yellow_duration=3,
        yellow_start_delay=0,
        min_green_steps=1,
        time_to_teleport=-1,
        additional_sumo_args=route_file_sumo_args((*demand_route_files.route_files, initial_population.route_file)),
        backend_kind=SumoBackendKind.TRACI,
    )
    initial_vehicle_count = 0
    post_warmup_vehicle_count = 0
    teleport_count = 0
    try:
        runtime.start()
        runtime.step()
        initial_vehicle_count = len(runtime.vehicle_api.getIDList())
        teleport_count += int(runtime.simulation_api.getStartingTeleportNumber())
        for _step in range(14):
            runtime.step()
            teleport_count += int(runtime.simulation_api.getStartingTeleportNumber())
        post_warmup_vehicle_count = len(runtime.vehicle_api.getIDList())
    finally:
        runtime.close()
        demand_route_files.cleanup()
        initial_population.cleanup()
    storage_capacity = lane_length_m / EFFECTIVE_VEHICLE_SPACING_M
    return InitialOccupancyProbe(
        target_occupancy=target_occupancy,
        requested_vehicle_count=initial_population.requested_vehicle_count,
        generated_vehicle_count=initial_population.generated_vehicle_count,
        realized_initial_occupancy=initial_vehicle_count / storage_capacity,
        post_warmup_vehicle_count=post_warmup_vehicle_count,
        teleport_count=teleport_count,
    )


def _static_record(
    scenario: ValidationScenario,
    controller_count: int,
    edge_count: int,
    lane_count: int,
    lane_length_m: float,
    route_flow_count: int,
    valid_route_count: int,
    boundary_source_count: int,
    boundary_destination_count: int,
    base_demand: float,
) -> GridValidationRecord:
    return GridValidationRecord(
        scenario_name=scenario.name,
        rows=scenario.rows,
        cols=scenario.cols,
        internal_junction_count=scenario.rows * scenario.cols,
        eligible_controller_count=len(controllable_node_ids(rows=scenario.rows, cols=scenario.cols)),
        controller_count=controller_count,
        signal_coverage=controller_count / len(controllable_node_ids(rows=scenario.rows, cols=scenario.cols)),
        edge_count=edge_count,
        lane_count=lane_count,
        lane_length_m=lane_length_m,
        route_flow_count=route_flow_count,
        valid_route_count=valid_route_count,
        boundary_source_count=boundary_source_count,
        boundary_destination_count=boundary_destination_count,
        base_demand_vehicles_per_hour=base_demand,
        target_initial_occupancy=None,
        requested_initial_vehicles=None,
        generated_initial_vehicles=None,
        realized_initial_occupancy=None,
        post_warmup_vehicle_count=None,
        warmup_seconds=None,
        warmup_teleport_count=None,
        demand_scale=None,
        policy=None,
        throughput_per_hour=None,
        completion_rate=None,
        completed_trip_waiting_time_s=None,
        switches_per_junction_per_minute=None,
        wait_density_s_per_m=None,
        evaluation_teleport_count=None,
    )


def _route_is_valid(network: Net, flow: ET.Element) -> bool:
    source = network.getEdge(flow.attrib['from'])
    destination = network.getEdge(flow.attrib['to'])
    route, _cost = network.getOptimalPath(source, destination, fastest=True)
    return route is not None and len(route) >= 2


def _edge_nodes(edge_id: str) -> tuple[str, str]:
    from_node, to_node = edge_id.split('_to_', 1)
    return from_node, to_node


def _plot_topology(
    scenario: ValidationScenario,
    scenario_directory: Path,
    output_path: Path,
) -> None:
    node_root = ET.parse(scenario_directory / 'grid.nod.xml').getroot()
    edge_root = ET.parse(scenario_directory / 'grid.edg.xml').getroot()
    node_positions = {
        node.attrib['id']: (float(node.attrib['x']), float(node.attrib['y'])) for node in node_root.findall('node')
    }
    signalized_nodes = frozenset(
        node.attrib['id'] for node in node_root.findall('node') if node.attrib.get('type') == 'traffic_light'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    for edge in edge_root.findall('edge'):
        start = node_positions[edge.attrib['from']]
        end = node_positions[edge.attrib['to']]
        axis.plot((start[0], end[0]), (start[1], end[1]), color='#a7b0b7', linewidth=0.7, zorder=1)
    for node_id, position in node_positions.items():
        if node_id.startswith('S_'):
            axis.scatter(*position, color='#8f9aa3', marker='s', s=18, zorder=2)
        elif node_id in signalized_nodes:
            axis.scatter(*position, color='#d62728', marker='o', s=55, zorder=3)
        else:
            axis.scatter(*position, color='#f0a202', marker='o', s=38, zorder=3)
    axis.set_title(_topology_title(scenario=scenario, signalized_count=len(signalized_nodes)))
    axis.set_aspect('equal')
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _topology_title(scenario: ValidationScenario, signalized_count: int) -> str:
    eligible_controller_count = len(controllable_node_ids(rows=scenario.rows, cols=scenario.cols))
    return f'{scenario.name}: {signalized_count}/{eligible_controller_count} eligible junctions signalized'


def validation_scenarios(suite: ValidationSuite) -> tuple[ValidationScenario, ...]:
    scenarios: list[ValidationScenario] = []
    if suite in (ValidationSuite.MATCHED, ValidationSuite.ALL):
        scenarios.extend(
            ValidationScenario(name=scenario.name, rows=scenario.rows, cols=scenario.cols)
            for scenario in MATCHED_GRID_SCENARIOS
        )
    if suite in (ValidationSuite.COVERAGE, ValidationSuite.ALL):
        scenarios.extend(
            ValidationScenario(name=scenario.name, rows=scenario.rows, cols=scenario.cols)
            for scenario in GRID_COVERAGE_SCENARIOS
        )
    return tuple(scenarios)


def run_validation_suite(
    scenarios: tuple[ValidationScenario, ...],
    configuration_root: Path,
    output_directory: Path,
    demand_scales: tuple[float, ...],
    simulation_seed: int,
    simulation_steps: int,
    skip_simulation: bool,
    workers: int,
) -> tuple[GridValidationRecord, ...]:
    if workers <= 0:
        raise ValueError('workers must be positive.')
    run_specs = tuple(
        ValidationRunSpec(
            scenario=scenario,
            configuration_root=configuration_root,
            output_directory=output_directory,
            demand_scales=demand_scales,
            simulation_seed=simulation_seed,
            simulation_steps=simulation_steps,
            skip_simulation=skip_simulation,
        )
        for scenario in scenarios
    )
    if workers == 1:
        record_groups = tuple(_run_validation_spec(run_spec) for run_spec in run_specs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            record_groups = tuple(executor.map(_run_validation_spec, run_specs))
    return tuple(record for record_group in record_groups for record in record_group)


def _run_validation_spec(run_spec: ValidationRunSpec) -> tuple[GridValidationRecord, ...]:
    return validate_scenario(
        scenario=run_spec.scenario,
        configuration_root=run_spec.configuration_root,
        output_directory=run_spec.output_directory,
        demand_scales=run_spec.demand_scales,
        simulation_seed=run_spec.simulation_seed,
        simulation_steps=run_spec.simulation_steps,
        skip_simulation=run_spec.skip_simulation,
    )


def write_report(output_directory: Path, records: tuple[GridValidationRecord, ...]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    report = GridValidationReport(records=records)
    (output_directory / 'validation.json').write_text(report.model_dump_json(indent=2), encoding='utf-8')
    fieldnames = tuple(GridValidationRecord.model_fields.keys())
    with (output_directory / 'validation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.model_dump() for record in records)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--configuration-root',
        type=Path,
        default=ROOT / 'configs' / 'grid_generalization',
    )
    parser.add_argument(
        '--output-directory',
        type=Path,
        default=ROOT / 'reports' / 'grid_generalization_validation',
    )
    parser.add_argument(
        '--suite',
        choices=tuple(suite.value for suite in ValidationSuite),
        default=ValidationSuite.ALL.value,
    )
    parser.add_argument('--demand-scales', nargs='+', type=float, default=(0.6, 0.7, 0.8))
    parser.add_argument('--simulation-seed', type=int, default=9101)
    parser.add_argument('--simulation-steps', type=int, default=1800)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument(
        '--scenario',
        action='append',
        default=[],
        help='Validate only the named scenario; may be repeated',
    )
    parser.add_argument('--skip-simulation', action='store_true')
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    scenarios = validation_scenarios(ValidationSuite(arguments.suite))
    if arguments.scenario:
        requested_scenarios = frozenset(arguments.scenario)
        scenarios = tuple(scenario for scenario in scenarios if scenario.name in requested_scenarios)
        missing_scenarios = requested_scenarios - frozenset(scenario.name for scenario in scenarios)
        if missing_scenarios:
            raise ValueError(f'Unknown validation scenarios: {", ".join(sorted(missing_scenarios))}')
    records = run_validation_suite(
        scenarios=scenarios,
        configuration_root=arguments.configuration_root,
        output_directory=arguments.output_directory,
        demand_scales=tuple(arguments.demand_scales),
        simulation_seed=arguments.simulation_seed,
        simulation_steps=arguments.simulation_steps,
        skip_simulation=arguments.skip_simulation,
        workers=arguments.workers,
    )
    write_report(output_directory=arguments.output_directory, records=records)
    print(f'Wrote {arguments.output_directory / "validation.json"}')
    print(f'Wrote {arguments.output_directory / "validation.csv"}')


if __name__ == '__main__':
    main()
