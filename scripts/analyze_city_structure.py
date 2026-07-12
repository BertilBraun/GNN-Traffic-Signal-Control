"""Extract reproducible structural statistics for the five paper city networks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path  # noqa: E402
from scripts.inspect_movement_city import InspectionReport, inspect_city_config  # noqa: E402


@dataclass(frozen=True)
class CityDefinition:
    city: str
    scenario: str
    role: str

    @property
    def sumo_config_path(self) -> Path:
        return ROOT / 'configs' / self.scenario / f'{self.scenario}.sumocfg'


@dataclass(frozen=True)
class NetworkXmlStatistics:
    junction_count: int
    signalized_junction_count: int
    external_edge_count: int
    lane_count: int
    lane_length_m: float


@dataclass(frozen=True)
class CityStructureStatistics:
    city: str
    scenario: str
    role: str
    junctions: int
    signalized_junctions: int
    traffic_light_controllers: int
    controllable_policy_junctions: int
    pass_through_signalized_junctions: int
    unsupported_signalized_junctions: int
    lane_groups: int
    movements: int
    unsignalized_lane_group_connectors: int
    input_lane_to_movement_edges: int
    output_lane_to_movement_edges: int
    movement_to_input_lane_edges: int
    movement_to_output_lane_edges: int
    typed_message_edges_total: int
    selectable_phases: int
    phases_per_junction_mean: float
    phases_per_junction_median: float
    phases_per_junction_min: int
    phases_per_junction_max: int
    external_edges: int
    lanes: int
    lane_length_km: float


CITIES = (
    CityDefinition('Karlsruhe', 'karlsruhe_oststadt', 'PPO rollout'),
    CityDefinition('Mannheim', 'mannheim_innenstadt', 'PPO rollout'),
    CityDefinition('Stuttgart', 'stuttgart_mitte', 'PPO rollout'),
    CityDefinition('Heidelberg', 'heidelberg_bergheim', 'PPO rollout'),
    CityDefinition('Freiburg', 'freiburg_altstadt', 'held out from PPO rollouts'),
)


def read_network_xml_statistics(network_path: Path) -> NetworkXmlStatistics:
    """Read topology and lane-length statistics directly from a SUMO network XML file."""
    root = ElementTree.parse(network_path).getroot()
    junctions = tuple(
        junction
        for junction in root.findall('junction')
        if not junction.attrib['id'].startswith(':') and junction.attrib.get('type') != 'internal'
    )
    external_edges = tuple(
        edge
        for edge in root.findall('edge')
        if not edge.attrib['id'].startswith(':') and edge.attrib.get('function') != 'internal'
    )
    lanes = tuple(lane for edge in external_edges for lane in edge.findall('lane'))
    return NetworkXmlStatistics(
        junction_count=len(junctions),
        signalized_junction_count=sum(
            junction.attrib.get('type', '').startswith('traffic_light') for junction in junctions
        ),
        external_edge_count=len(external_edges),
        lane_count=len(lanes),
        lane_length_m=sum(float(lane.attrib['length']) for lane in lanes),
    )


def combine_statistics(
    city: CityDefinition,
    inspection: InspectionReport,
    network: NetworkXmlStatistics,
) -> CityStructureStatistics:
    """Combine runtime graph extraction with static SUMO network statistics."""
    controllable_phase_counts = tuple(
        phase_count for _, phase_count in inspection.phase_counts_by_traffic_light if phase_count > 1
    )
    if not controllable_phase_counts:
        raise ValueError(f'{city.scenario} has no controllable traffic lights')
    controllable_count = len(controllable_phase_counts)
    typed_edge_count = inspection.movement_count
    return CityStructureStatistics(
        city=city.city,
        scenario=city.scenario,
        role=city.role,
        junctions=network.junction_count,
        signalized_junctions=network.signalized_junction_count,
        traffic_light_controllers=inspection.traffic_light_count,
        controllable_policy_junctions=controllable_count,
        pass_through_signalized_junctions=inspection.pass_through_traffic_light_count,
        unsupported_signalized_junctions=len(inspection.skipped_traffic_lights),
        lane_groups=inspection.lane_group_count,
        movements=inspection.movement_count,
        unsignalized_lane_group_connectors=inspection.lane_lane_connector_count,
        input_lane_to_movement_edges=typed_edge_count,
        output_lane_to_movement_edges=typed_edge_count,
        movement_to_input_lane_edges=typed_edge_count,
        movement_to_output_lane_edges=typed_edge_count,
        typed_message_edges_total=4 * typed_edge_count,
        selectable_phases=sum(controllable_phase_counts),
        phases_per_junction_mean=statistics.fmean(controllable_phase_counts),
        phases_per_junction_median=statistics.median(controllable_phase_counts),
        phases_per_junction_min=min(controllable_phase_counts),
        phases_per_junction_max=max(controllable_phase_counts),
        external_edges=network.external_edge_count,
        lanes=network.lane_count,
        lane_length_km=network.lane_length_m / 1000.0,
    )


def analyze_city(city: CityDefinition, seed: int) -> CityStructureStatistics:
    """Extract one city's statistics using the same graph construction as runtime control."""
    inspection = inspect_city_config(city.sumo_config_path, seed=seed, time_to_teleport=-1)
    network_path = resolve_sumocfg_net_path(city.sumo_config_path)
    network = read_network_xml_statistics(network_path)
    return combine_statistics(city=city, inspection=inspection, network=network)


def write_csv(statistics_rows: tuple[CityStructureStatistics, ...], output_path: Path) -> None:
    """Write machine-readable city statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(asdict(statistics_rows[0]))
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(statistics_row) for statistics_row in statistics_rows)


def write_markdown(statistics_rows: tuple[CityStructureStatistics, ...], output_path: Path) -> None:
    """Write a concise paper-oriented table and definitions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Structural statistics for the five city scenarios',
        '',
        '| City | Role | Junctions | Signalized nodes | Policy controllers | LaneGroups | Movements | Unsignalized connectors | Selectable phases | Phases/controller (mean; median; range) | Typed message edges | Lane length (km) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in statistics_rows:
        lines.append(
            f'| {row.city} | {row.role} | {row.junctions} | {row.signalized_junctions} | '
            f'{row.controllable_policy_junctions} | {row.lane_groups} | {row.movements} | '
            f'{row.unsignalized_lane_group_connectors} | {row.selectable_phases} | '
            f'{row.phases_per_junction_mean:.2f}; {row.phases_per_junction_median:.1f}; '
            f'{row.phases_per_junction_min}-{row.phases_per_junction_max} | '
            f'{row.typed_message_edges_total} | {row.lane_length_km:.2f} |'
        )
    lines.extend(
        [
            '',
            '## Definitions and reproducibility',
            '',
            '- `Junctions` counts non-internal `<junction>` elements in the saved SUMO network.',
            '- `Signalized` counts those junctions whose SUMO type begins with `traffic_light`.',
            '- `Policy controllers` counts extracted traffic-light programs with more than one selectable phase; these are the independent policy action sites.',
            '- Signalized-node and controller counts can differ because SUMO controller structures may represent clustered or joined junctions.',
            '- LaneGroups, Movements, connectors, and phase incidences are rebuilt through the current runtime graph extraction.',
            '- Each Movement contributes one edge to each of the four typed LaneGroup/Movement relations.',
            '- `Lane length` sums the lengths of all lanes on non-internal SUMO edges; it is lane-kilometres, not centreline road length.',
            '- Full-precision values and controller/pass-through/unsupported counts are in `city_structure_statistics.csv`.',
            '',
            'Generated with:',
            '',
            '```powershell',
            'uv run python scripts\\analyze_city_structure.py',
            '```',
            '',
        ]
    )
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='SUMO startup seed; static extraction is deterministic')
    parser.add_argument(
        '--csv-output',
        type=Path,
        default=ROOT / 'docs' / 'results' / 'city_structure_statistics.csv',
    )
    parser.add_argument(
        '--markdown-output',
        type=Path,
        default=ROOT / 'docs' / 'results' / 'city_structure_statistics.md',
    )
    return parser.parse_args()


def main() -> None:
    """Analyze all paper cities and write reproducible outputs."""
    arguments = parse_arguments()
    statistics_rows = tuple(analyze_city(city, arguments.seed) for city in CITIES)
    write_csv(statistics_rows, arguments.csv_output)
    write_markdown(statistics_rows, arguments.markdown_output)
    print(f'Wrote {arguments.csv_output}')
    print(f'Wrote {arguments.markdown_output}')


if __name__ == '__main__':
    main()
