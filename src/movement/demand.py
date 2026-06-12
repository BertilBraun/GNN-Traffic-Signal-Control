"""Temporary demand transformations for SUMO route files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class DemandRouteFiles:
    route_files: tuple[Path, ...]
    temporary_directory: tempfile.TemporaryDirectory[str] | None

    def cleanup(self) -> None:
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


def route_files_for_demand_scale(
    cfg_path: str | Path,
    demand_scale: float,
) -> DemandRouteFiles:
    """Return route files for a scaled demand run."""
    if demand_scale <= 0.0:
        raise ValueError('demand_scale must be positive.')
    route_files = resolve_sumocfg_route_files(cfg_path)
    if demand_scale == 1.0:
        return DemandRouteFiles(
            route_files=route_files,
            temporary_directory=None,
        )
    temporary_directory = tempfile.TemporaryDirectory(prefix='movement_demand_')
    output_dir = Path(temporary_directory.name)
    scaled_route_files = tuple(
        _write_scaled_route_file(
            source_path=route_file,
            output_path=output_dir / route_file.name,
            demand_scale=demand_scale,
        )
        for route_file in route_files
    )
    return DemandRouteFiles(
        route_files=scaled_route_files,
        temporary_directory=temporary_directory,
    )


def route_file_sumo_args(route_files: Sequence[Path]) -> tuple[str, str]:
    """Build SUMO command args for explicit route files."""
    return (
        '--route-files',
        ','.join(str(route_file) for route_file in route_files),
    )


def resolve_sumocfg_route_files(cfg_path: str | Path) -> tuple[Path, ...]:
    """Resolve route files referenced by a SUMO config."""
    config_path = Path(cfg_path)
    root = ET.parse(config_path).getroot()
    route_files = root.find('./input/route-files')
    if route_files is None or 'value' not in route_files.attrib:
        raise ValueError(f'{config_path} does not define input/route-files.')
    resolved_paths = []
    for route_file in route_files.attrib['value'].split(','):
        route_path = Path(route_file.strip())
        if not route_path.is_absolute():
            route_path = config_path.parent / route_path
        resolved_paths.append(route_path)
    return tuple(resolved_paths)


def _write_scaled_route_file(
    source_path: Path,
    output_path: Path,
    demand_scale: float,
) -> Path:
    tree = ET.parse(source_path)
    root = tree.getroot()
    for flow in root.findall('.//flow'):
        _scale_float_attribute(flow, 'probability', demand_scale)
        _scale_float_attribute(flow, 'vehsPerHour', demand_scale)
        _scale_period_attribute(flow, demand_scale)
        _scale_integer_attribute(flow, 'number', demand_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return output_path


def _scale_float_attribute(
    element: ET.Element,
    attribute_name: str,
    demand_scale: float,
) -> None:
    raw_value = element.attrib.get(attribute_name)
    if raw_value is None:
        return
    scaled_value = float(raw_value) * demand_scale
    element.set(attribute_name, f'{scaled_value:.6g}')


def _scale_period_attribute(
    element: ET.Element,
    demand_scale: float,
) -> None:
    raw_value = element.attrib.get('period')
    if raw_value is None:
        return
    scaled_period = float(raw_value) / demand_scale
    element.set('period', f'{scaled_period:.6g}')


def _scale_integer_attribute(
    element: ET.Element,
    attribute_name: str,
    demand_scale: float,
) -> None:
    raw_value = element.attrib.get(attribute_name)
    if raw_value is None:
        return
    scaled_value = max(1, round(float(raw_value) * demand_scale))
    element.set(attribute_name, str(scaled_value))
