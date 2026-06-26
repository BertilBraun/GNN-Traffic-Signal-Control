"""Randomized initial traffic populations for SUMO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tempfile
import xml.etree.ElementTree as ET

import sumolib
from sumolib.net import Net
from sumolib.net.edge import Edge

from src.movement.demand import resolve_sumocfg_route_files

EFFECTIVE_VEHICLE_SPACING_M = 8.0
MAX_ROUTE_ATTEMPTS_PER_VEHICLE = 50
INITIAL_TRAFFIC_VEHICLE_CLASS = 'passenger'


@dataclass(frozen=True)
class InitialTrafficPopulation:
    route_file: Path
    target_occupancy: float
    requested_vehicle_count: int
    generated_vehicle_count: int
    temporary_directory: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


def generate_initial_traffic_population(
    cfg_path: str | Path,
    net_path: str | Path,
    target_occupancy: float,
    seed: int,
) -> InitialTrafficPopulation:
    """Generate vehicles distributed across valid route suffixes at time zero."""
    if target_occupancy < 0.0 or target_occupancy > 1.0:
        raise ValueError('target_occupancy must be between 0 and 1.')
    network = sumolib.net.readNet(str(net_path), withConnections=True)
    candidate_edges = _candidate_edges(network)
    destination_edges = _destination_edges(
        network=network,
        route_files=resolve_sumocfg_route_files(cfg_path),
    )
    requested_vehicle_count = round(target_occupancy * sum(_edge_storage_capacity(edge) for edge in candidate_edges))
    random_generator = random.Random(seed)
    vehicle_routes = _sample_vehicle_routes(
        network=network,
        candidate_edges=candidate_edges,
        destination_edges=destination_edges,
        vehicle_count=requested_vehicle_count,
        random_generator=random_generator,
    )
    temporary_directory = tempfile.TemporaryDirectory(prefix='movement_initial_traffic_')
    route_file = Path(temporary_directory.name) / 'initial_population.rou.xml'
    _write_population_route_file(route_file, vehicle_routes)
    return InitialTrafficPopulation(
        route_file=route_file,
        target_occupancy=target_occupancy,
        requested_vehicle_count=requested_vehicle_count,
        generated_vehicle_count=len(vehicle_routes),
        temporary_directory=temporary_directory,
    )


def sample_target_occupancy(
    minimum_occupancy: float,
    maximum_occupancy: float,
    seed: int,
) -> float:
    """Sample one deterministic rollout occupancy from an inclusive range."""
    if minimum_occupancy < 0.0 or maximum_occupancy > 1.0:
        raise ValueError('Initial occupancy bounds must be between 0 and 1.')
    if minimum_occupancy > maximum_occupancy:
        raise ValueError('minimum_occupancy must not exceed maximum_occupancy.')
    return random.Random(seed).uniform(minimum_occupancy, maximum_occupancy)


def _candidate_edges(network: Net) -> tuple[Edge, ...]:
    edges = tuple(
        edge
        for edge in network.getEdges()
        if not edge.getID().startswith(':')
        and edge.getFunction() == ''
        and edge.getLength() > 0.0
        and edge.getLaneNumber() > 0
        and edge.allows(INITIAL_TRAFFIC_VEHICLE_CLASS)
    )
    if not edges:
        raise ValueError('Network has no edges eligible for initial traffic.')
    return edges


def _destination_edges(
    network: Net,
    route_files: tuple[Path, ...],
) -> tuple[Edge, ...]:
    destination_ids: set[str] = set()
    for route_file in route_files:
        root = ET.parse(route_file).getroot()
        route_destinations = {
            route.attrib['id']: route.attrib['edges'].split()[-1]
            for route in root.findall('.//route')
            if 'id' in route.attrib and route.attrib.get('edges')
        }
        for flow in root.findall('.//flow'):
            destination_id = flow.attrib.get('to')
            if destination_id is not None:
                destination_ids.add(destination_id)
            route_id = flow.attrib.get('route')
            if route_id in route_destinations:
                destination_ids.add(route_destinations[route_id])
        for trip in root.findall('.//trip'):
            destination_id = trip.attrib.get('to')
            if destination_id is not None:
                destination_ids.add(destination_id)
        for vehicle in root.findall('.//vehicle'):
            route_id = vehicle.attrib.get('route')
            if route_id in route_destinations:
                destination_ids.add(route_destinations[route_id])
            inline_route = vehicle.find('route')
            if inline_route is not None and inline_route.attrib.get('edges'):
                destination_ids.add(inline_route.attrib['edges'].split()[-1])
    if not destination_ids:
        destination_ids.update(edge.getID() for edge in _candidate_edges(network) if not edge.getOutgoing())
    destinations = tuple(
        network.getEdge(edge_id)
        for edge_id in sorted(destination_ids)
        if network.getEdge(edge_id).allows(INITIAL_TRAFFIC_VEHICLE_CLASS)
    )
    if not destinations:
        raise ValueError('Route files do not define boundary destination edges.')
    return destinations


def _edge_storage_capacity(edge: Edge) -> float:
    return edge.getLength() * edge.getLaneNumber() / EFFECTIVE_VEHICLE_SPACING_M


def _sample_vehicle_routes(
    network: Net,
    candidate_edges: tuple[Edge, ...],
    destination_edges: tuple[Edge, ...],
    vehicle_count: int,
    random_generator: random.Random,
) -> tuple[tuple[str, ...], ...]:
    edge_weights = tuple(_edge_storage_capacity(edge) for edge in candidate_edges)
    routes: list[tuple[str, ...]] = []
    for _vehicle_index in range(vehicle_count):
        route = _sample_route(
            network=network,
            candidate_edges=candidate_edges,
            edge_weights=edge_weights,
            destination_edges=destination_edges,
            random_generator=random_generator,
        )
        if route is not None:
            routes.append(route)
    return tuple(routes)


def _sample_route(
    network: Net,
    candidate_edges: tuple[Edge, ...],
    edge_weights: tuple[float, ...],
    destination_edges: tuple[Edge, ...],
    random_generator: random.Random,
) -> tuple[str, ...] | None:
    for _attempt in range(MAX_ROUTE_ATTEMPTS_PER_VEHICLE):
        start_edge = random_generator.choices(
            candidate_edges,
            weights=edge_weights,
            k=1,
        )[0]
        destination_edge = random_generator.choice(destination_edges)
        if start_edge == destination_edge:
            continue
        route, _cost = network.getOptimalPath(
            start_edge,
            destination_edge,
            fastest=False,
            vClass=INITIAL_TRAFFIC_VEHICLE_CLASS,
        )
        if route is None or len(route) < 2:
            continue
        return tuple(edge.getID() for edge in route)
    return None


def _write_population_route_file(
    path: Path,
    vehicle_routes: tuple[tuple[str, ...], ...],
) -> None:
    root = ET.Element('routes')
    ET.SubElement(
        root,
        'vType',
        {
            'id': 'initial_car',
            'accel': '2.6',
            'decel': '4.5',
            'length': '5.0',
            'minGap': '2.5',
            'maxSpeed': '13.89',
        },
    )
    for vehicle_index, route in enumerate(vehicle_routes):
        vehicle = ET.SubElement(
            root,
            'vehicle',
            {
                'id': f'initial_{vehicle_index}',
                'type': 'initial_car',
                'depart': '0',
                'departLane': 'best',
                'departPos': 'random_free',
                'departSpeed': 'avg',
            },
        )
        ET.SubElement(vehicle, 'route', {'edges': ' '.join(route)})
    ET.ElementTree(root).write(
        path,
        encoding='utf-8',
        xml_declaration=True,
    )
