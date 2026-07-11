"""Typed experiment configuration for multi-city training runs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
import yaml


class CitySplit(str, Enum):
    TRAIN = 'train'
    HELD_OUT = 'held_out'


class ExperimentEvaluationPolicy(str, Enum):
    LEARNED = 'learned'
    MAX_PRESSURE = 'max-pressure'
    QUEUE = 'queue'


class ExperimentPpoRewardMode(str, Enum):
    DELAY_DENSITY = 'delay-density'
    THROUGHPUT = 'throughput'


class ExperimentLearnedEvaluationActionMode(str, Enum):
    DETERMINISTIC = 'deterministic'
    SAMPLE = 'sample'


class ExperimentCityConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    split: CitySplit
    sumo_config: Path
    build_config: Path
    rollout_jobs_per_iteration: int = Field(
        ge=0,
        validation_alias=AliasChoices('rollout_jobs_per_iteration', 'rollout_workers'),
    )
    rollout_priority: int = Field(ge=0, default=0)
    minimum_train_scale: float | None = Field(gt=0.0, default=None, alias='train_scale_min')
    maximum_train_scale: float | None = Field(gt=0.0, default=None, alias='train_scale_max')

    @property
    def rollout_workers(self) -> int:
        return self.rollout_jobs_per_iteration

    @model_validator(mode='after')
    def validate_rollout_workers_for_split(self) -> 'ExperimentCityConfiguration':
        match self.split:
            case CitySplit.TRAIN:
                pass
            case CitySplit.HELD_OUT:
                if self.rollout_jobs_per_iteration != 0:
                    raise ValueError(f'held-out city {self.name} must define rollout_jobs_per_iteration: 0')
        return self

    @model_validator(mode='after')
    def validate_city_demand_scale_range(self) -> 'ExperimentCityConfiguration':
        if (
            self.minimum_train_scale is not None
            and self.maximum_train_scale is not None
            and self.minimum_train_scale > self.maximum_train_scale
        ):
            raise ValueError(f'city {self.name} train_scale_min must be less than or equal to train_scale_max')
        return self


class ExperimentSimulationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    decision_interval: int = Field(gt=0)
    time_to_teleport: int
    yellow_duration: int = Field(ge=0)
    minimum_green_steps: int = Field(gt=0, alias='min_green_steps')
    minimum_initial_occupancy: float = Field(ge=0.0, alias='initial_occupancy_min')
    maximum_initial_occupancy: float = Field(ge=0.0, alias='initial_occupancy_max')

    @model_validator(mode='after')
    def validate_initial_occupancy_range(self) -> 'ExperimentSimulationConfiguration':
        if self.minimum_initial_occupancy > self.maximum_initial_occupancy:
            raise ValueError('initial_occupancy_min must be less than or equal to initial_occupancy_max')
        return self


class ExperimentDemandConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    minimum_train_scale: float = Field(gt=0.0, alias='train_scale_min')
    maximum_train_scale: float = Field(gt=0.0, alias='train_scale_max')
    evaluation_scales: tuple[float, ...] = Field(min_length=1, alias='eval_scales')

    @model_validator(mode='after')
    def validate_demand_scales(self) -> 'ExperimentDemandConfiguration':
        if self.minimum_train_scale > self.maximum_train_scale:
            raise ValueError('train_scale_min must be less than or equal to train_scale_max')
        for demand_scale in self.evaluation_scales:
            if demand_scale <= 0.0:
                raise ValueError('eval_scales must contain only positive demand scales')
        return self


class ExperimentImitationLearningConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples_per_city: int = Field(gt=0)
    samples_per_simulation: int = Field(gt=0)
    collection_workers: int = Field(gt=0)
    epochs: int = Field(gt=0)
    samples_per_batch: int = Field(gt=0)
    phase_loss_coefficient: float = Field(ge=0.0)


class ExperimentProximalPolicyOptimizationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    iterations: int = Field(gt=0)
    steps_per_rollout: int = Field(gt=0)
    rollouts_per_update: int = Field(gt=0)
    rollout_workers: int = Field(gt=0)
    update_epochs: int = Field(gt=0, default=4)
    value_warmup_iterations: int = Field(ge=0, default=20)
    warmup_epochs: int = Field(gt=0, default=8)
    transitions_per_batch: int = Field(gt=0, default=32)
    update_batch_workers: int = Field(ge=0, default=0)
    reward_sample_interval: int = Field(gt=0, default=5)
    reward_mode: ExperimentPpoRewardMode = ExperimentPpoRewardMode.DELAY_DENSITY
    global_reward_weight: float = Field(ge=0.0, default=0.1)
    flow_reward_weight: float = Field(ge=0.0, default=0.1)
    throughput_reward_weight: float = Field(ge=0.0, default=1.0)
    progress_reward_weight: float = Field(ge=0.0, default=0.03)
    gridlock_penalty_weight: float = Field(ge=0.0, default=0.02)
    speed_change_weight: float = Field(ge=0.0, default=0.02)
    evaluate_every_iterations: int = Field(ge=0, alias='eval_every_iterations')
    evaluation_workers: int = Field(gt=0, default=1, alias='eval_workers')
    evaluation_learned_device: str = Field(min_length=1, default='cpu', alias='eval_learned_device')
    evaluation_learned_action_mode: ExperimentLearnedEvaluationActionMode = Field(
        default=ExperimentLearnedEvaluationActionMode.DETERMINISTIC,
        alias='eval_learned_action_mode',
    )
    evaluation_learned_temperature: float = Field(gt=0.0, default=1.0, alias='eval_learned_temperature')
    save_every_iterations: int = Field(gt=0)


class ExperimentEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    policies: tuple[ExperimentEvaluationPolicy, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    steps: int = Field(gt=0)


class ExperimentConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    cities: tuple[ExperimentCityConfiguration, ...] = Field(min_length=1)
    simulation: ExperimentSimulationConfiguration
    demand: ExperimentDemandConfiguration
    imitation_learning: ExperimentImitationLearningConfiguration
    proximal_policy_optimization: ExperimentProximalPolicyOptimizationConfiguration = Field(alias='ppo')
    evaluation: ExperimentEvaluationConfiguration

    @model_validator(mode='after')
    def validate_city_splits(self) -> 'ExperimentConfiguration':
        city_names = tuple(city.name for city in self.cities)
        duplicate_names = tuple(name for name in dict.fromkeys(city_names) if city_names.count(name) > 1)
        if duplicate_names:
            raise ValueError(f'duplicate city names are not allowed: {", ".join(duplicate_names)}')

        held_out_cities = tuple(city.name for city in self.cities if city.split == CitySplit.HELD_OUT)
        if len(held_out_cities) != 1:
            raise ValueError(f'exactly one held-out city is required, found {len(held_out_cities)}')
        if self.proximal_policy_optimization.reward_sample_interval > self.simulation.decision_interval:
            raise ValueError('ppo.reward_sample_interval must not exceed simulation.decision_interval')
        return self

    @property
    def train_cities(self) -> tuple[ExperimentCityConfiguration, ...]:
        return tuple(city for city in self.cities if city.split == CitySplit.TRAIN)

    @property
    def held_out_city(self) -> ExperimentCityConfiguration:
        held_out_cities = tuple(city for city in self.cities if city.split == CitySplit.HELD_OUT)
        if len(held_out_cities) != 1:
            raise ValueError(f'exactly one held-out city is required, found {len(held_out_cities)}')
        return held_out_cities[0]


def load_experiment_configuration(configuration_path: Path, project_root: Path) -> ExperimentConfiguration:
    if not configuration_path.exists():
        raise ValueError(f'experiment configuration file does not exist: {configuration_path}')
    payload = yaml.safe_load(configuration_path.read_text(encoding='utf-8-sig'))
    configuration = ExperimentConfiguration.model_validate(payload)
    _validate_referenced_files(configuration=configuration, project_root=project_root)
    return configuration


def resolve_experiment_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def _validate_referenced_files(configuration: ExperimentConfiguration, project_root: Path) -> None:
    missing_paths: list[Path] = []
    for city in configuration.cities:
        sumo_config_path = resolve_experiment_path(path=city.sumo_config, project_root=project_root)
        build_config_path = resolve_experiment_path(path=city.build_config, project_root=project_root)
        if not sumo_config_path.exists():
            missing_paths.append(sumo_config_path)
        if not build_config_path.exists():
            missing_paths.append(build_config_path)

    if missing_paths:
        missing_path_text = ', '.join(str(path) for path in missing_paths)
        raise ValueError(f'experiment configuration references missing files: {missing_path_text}')
