from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import sumolib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.initial_traffic import (
    generate_initial_traffic_population,
    sample_target_occupancy,
)


def test_sample_target_occupancy_is_seeded_and_bounded() -> None:
    first = sample_target_occupancy(
        minimum_occupancy=0.1,
        maximum_occupancy=0.3,
        seed=42,
    )
    second = sample_target_occupancy(
        minimum_occupancy=0.1,
        maximum_occupancy=0.3,
        seed=42,
    )

    assert first == second
    assert 0.1 <= first <= 0.3


def test_generate_initial_population_writes_valid_route_suffixes() -> None:
    population = generate_initial_traffic_population(
        cfg_path=ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.sumocfg',
        net_path=ROOT / 'configs' / 'grid_3x3_dedicated' / 'grid.net.xml',
        target_occupancy=0.01,
        seed=7,
    )

    try:
        root = ET.parse(population.route_file).getroot()
        vehicles = root.findall('vehicle')

        assert vehicles
        assert len(vehicles) == population.generated_vehicle_count
        assert all(vehicle.attrib['depart'] == '0' for vehicle in vehicles)
        assert all(vehicle.attrib['departPos'] == 'random_free' for vehicle in vehicles)
        assert all(len(vehicle.find('route').attrib['edges'].split()) >= 2 for vehicle in vehicles)
    finally:
        population.cleanup()


def test_generate_city_initial_population_uses_passenger_valid_routes() -> None:
    population = generate_initial_traffic_population(
        cfg_path=ROOT / 'configs' / 'mannheim_innenstadt' / 'mannheim_innenstadt.sumocfg',
        net_path=ROOT / 'configs' / 'mannheim_innenstadt' / 'mannheim_innenstadt.net.xml',
        target_occupancy=0.06,
        seed=100,
    )

    try:
        network = sumolib.net.readNet(str(ROOT / 'configs' / 'mannheim_innenstadt' / 'mannheim_innenstadt.net.xml'))
        root = ET.parse(population.route_file).getroot()
        routes = [vehicle.find('route').attrib['edges'].split() for vehicle in root.findall('vehicle')]

        assert routes
        assert all(network.getEdge(edge_id).allows('passenger') for route in routes for edge_id in route)
    finally:
        population.cleanup()
