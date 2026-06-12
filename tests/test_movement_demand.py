from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.demand import route_file_sumo_args, route_files_for_demand_scale


def test_route_files_for_demand_scale_writes_scaled_temporary_route_file(tmp_path: Path) -> None:
    route_path = tmp_path / 'grid.rou.xml'
    route_path.write_text(
        (
            '<routes>'
            '<flow id="prob" probability="0.0300"/>'
            '<flow id="hour" vehsPerHour="900"/>'
            '<flow id="period" period="4"/>'
            '</routes>'
        ),
        encoding='utf-8',
    )
    config_path = tmp_path / 'grid.sumocfg'
    config_path.write_text(
        '<configuration><input><route-files value="grid.rou.xml"/></input></configuration>',
        encoding='utf-8',
    )

    demand_route_files = route_files_for_demand_scale(
        cfg_path=config_path,
        demand_scale=0.5,
    )

    try:
        scaled_path = demand_route_files.route_files[0]
        root = ET.parse(scaled_path).getroot()
        flows = {flow.attrib['id']: flow for flow in root.findall('flow')}

        assert flows['prob'].attrib['probability'] == '0.015'
        assert flows['hour'].attrib['vehsPerHour'] == '450'
        assert flows['period'].attrib['period'] == '8'
        assert route_file_sumo_args(demand_route_files.route_files) == (
            '--route-files',
            str(scaled_path),
        )
    finally:
        demand_route_files.cleanup()
