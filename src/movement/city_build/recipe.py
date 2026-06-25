"""Typed network-workbench build recipes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator
import yaml


class OsmCachePolicy(str, Enum):
    REUSE = 'reuse'
    REFRESH = 'refresh'


class SourceRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    bbox: str | None = None
    osm: Path | None = None
    cache_policy: OsmCachePolicy = OsmCachePolicy.REUSE

    @model_validator(mode='after')
    def validate_single_source(self) -> 'SourceRecipe':
        if (self.bbox is None) == (self.osm is None):
            raise ValueError('source must define exactly one of bbox or osm')
        return self


class NetconvertRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    join_dist: float = 40.0
    promote_all_junctions_to_tl: bool = False


class DemandRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    route_count: int = 300
    demand_vehicles_per_hour: float = 900.0


class VerificationRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    inspect: bool = True
    movement_graph_html: bool = True
    detection_html: bool = False
    gui: bool = True
    gui_steps: int = 1800
    demand_scale: float = 1.0
    time_to_teleport: int = -1
    detection_steps: int = 120
    detection_sample_every: int = 1


class EvaluationRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    policies: tuple[str, ...] = ('max-pressure', 'queue')
    seeds: tuple[int, ...] = (42,)
    steps: int = 1800


class CityBuildRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    source: SourceRecipe
    netconvert: NetconvertRecipe = NetconvertRecipe()
    demand: DemandRecipe = DemandRecipe()
    verification: VerificationRecipe = VerificationRecipe()
    evaluation: EvaluationRecipe = EvaluationRecipe()


@dataclass(frozen=True)
class CityBuildPaths:
    recipe_path: Path
    city_directory: Path
    reports_directory: Path
    osm_path: Path
    net_path: Path
    prune_path: Path
    traffic_light_path: Path
    route_path: Path
    additional_path: Path
    sumo_config_path: Path
    inspection_report_path: Path
    movement_graph_path: Path
    movement_detection_path: Path
    build_summary_path: Path
    evaluation_directory: Path


def load_build_recipe(recipe_path: Path) -> CityBuildRecipe:
    payload = yaml.safe_load(recipe_path.read_text(encoding='utf-8-sig'))
    return CityBuildRecipe.model_validate(payload)


def save_build_recipe(recipe_path: Path, recipe: CityBuildRecipe) -> None:
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    payload = recipe.model_dump(mode='json', exclude_none=True)
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def city_build_paths(recipe_path: Path, recipe: CityBuildRecipe) -> CityBuildPaths:
    city_directory = recipe_path.parent
    reports_directory = city_directory / 'reports'
    base_name = recipe.name
    return CityBuildPaths(
        recipe_path=recipe_path,
        city_directory=city_directory,
        reports_directory=reports_directory,
        osm_path=city_directory / f'{base_name}.osm',
        net_path=city_directory / f'{base_name}.net.xml',
        prune_path=city_directory / f'{base_name}.prune.json',
        traffic_light_path=city_directory / f'{base_name}.tll.xml',
        route_path=city_directory / f'{base_name}.rou.xml',
        additional_path=city_directory / f'{base_name}.add.xml',
        sumo_config_path=city_directory / f'{base_name}.sumocfg',
        inspection_report_path=reports_directory / 'inspection.txt',
        movement_graph_path=reports_directory / 'movement_graph.html',
        movement_detection_path=reports_directory / 'movement_detection.html',
        build_summary_path=reports_directory / 'build_summary.json',
        evaluation_directory=reports_directory / 'evaluation',
    )
