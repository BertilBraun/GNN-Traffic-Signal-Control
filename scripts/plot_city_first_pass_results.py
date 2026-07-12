"""Generate the static figures for the city-first-pass PPO results report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_DIRECTORY = (
    PROJECT_ROOT
    / 'artifacts'
    / 'ppo_runs'
    / 'city_first_pass_throughput_progress_025_sample_eval_v3'
    / 'tensorboard_full'
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / 'docs' / 'results' / 'assets'
SELECTED_ITERATION = 85


@dataclass(frozen=True)
class City:
    key: str
    label: str
    split: str


@dataclass(frozen=True)
class Series:
    steps: tuple[int, ...]
    values: tuple[float, ...]


CITIES = (
    City('karlsruhe_oststadt', 'Karlsruhe', 'train'),
    City('mannheim_innenstadt', 'Mannheim', 'train'),
    City('stuttgart_mitte', 'Stuttgart', 'train'),
    City('heidelberg_bergheim', 'Heidelberg', 'train'),
    City('freiburg_altstadt', 'Freiburg (validation)', 'held_out'),
)
POLICY_LABELS = (
    ('learned', 'Learned (sampled)'),
    ('max-pressure', 'Max pressure'),
    ('queue', 'Queue'),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-dir', type=Path, default=DEFAULT_EVENT_DIRECTORY)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def load_events(event_directory: Path) -> EventAccumulator:
    event_paths = tuple(event_directory.glob('events.out.tfevents.*'))
    if len(event_paths) != 1:
        raise ValueError(f'Expected one TensorBoard event file in {event_directory}, found {len(event_paths)}.')
    accumulator = EventAccumulator(str(event_paths[0]), size_guidance={'scalars': 0})
    accumulator.Reload()
    return accumulator


def scalar_series(accumulator: EventAccumulator, tag: str) -> Series:
    events: list[ScalarEvent] = accumulator.Scalars(tag)
    return Series(
        steps=tuple(event.step for event in events),
        values=tuple(event.value for event in events),
    )


def evaluation_tag(city: City, policy: str, metric: str) -> str:
    return f'eval/{city.split}/{city.key}/{policy}/demand_1_000/{metric}'


def baseline_value(accumulator: EventAccumulator, city: City, policy: str, metric: str) -> float:
    series = scalar_series(accumulator, evaluation_tag(city, policy, metric))
    if not series.values:
        raise ValueError(f'Missing baseline series for {city.key}/{policy}/{metric}.')
    return series.values[0]


def configure_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 120,
            'savefig.dpi': 220,
            'font.size': 10,
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.frameon': False,
        }
    )


def mark_selected_iteration(axis: Axes) -> None:
    axis.axvline(SELECTED_ITERATION, color='#222222', linestyle=':', linewidth=1.4)


def save_figure(figure: Figure, output_path: Path) -> None:
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def plot_city_throughput(accumulator: EventAccumulator, output_directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    colors = plt.get_cmap('tab10').colors
    for index, city in enumerate(CITIES):
        series = scalar_series(accumulator, evaluation_tag(city, 'learned', 'throughput_per_hour'))
        axis.plot(
            series.steps,
            series.values,
            marker='o',
            markersize=3.2,
            linewidth=1.8,
            color=colors[index],
            label=city.label,
        )
    mark_selected_iteration(axis)
    axis.annotate(
        'selected checkpoint\niteration 85',
        xy=(SELECTED_ITERATION, axis.get_ylim()[1]),
        xytext=(8, -8),
        textcoords='offset points',
        ha='left',
        va='top',
    )
    axis.set(
        title='Learned-policy throughput across the full training run',
        xlabel='PPO iteration',
        ylabel='Throughput (vehicles/hour)',
    )
    axis.legend(ncols=2)
    save_figure(figure, output_directory / 'learned-throughput-full-run.png')


def plot_city_comparison(accumulator: EventAccumulator, output_directory: Path, city: City) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    learned = scalar_series(accumulator, evaluation_tag(city, 'learned', 'throughput_per_hour'))
    axis.plot(
        learned.steps,
        learned.values,
        marker='o',
        markersize=3.3,
        linewidth=2.0,
        color='#276FBF',
        label='Learned (sampled)',
    )
    axis.axhline(
        baseline_value(accumulator, city, 'max-pressure', 'throughput_per_hour'),
        color='#C44900',
        linewidth=1.8,
        label='Max pressure',
    )
    axis.axhline(
        baseline_value(accumulator, city, 'queue', 'throughput_per_hour'),
        color='#4C956C',
        linewidth=1.8,
        linestyle='--',
        label='Queue',
    )
    mark_selected_iteration(axis)
    selected_index = learned.steps.index(SELECTED_ITERATION)
    axis.scatter(
        (SELECTED_ITERATION,),
        (learned.values[selected_index],),
        color='#222222',
        marker='D',
        s=42,
        zorder=5,
        label='Selected checkpoint',
    )
    axis.set(
        title=f'{city.label}: learned throughput and fixed baselines',
        xlabel='PPO iteration',
        ylabel='Throughput (vehicles/hour)',
    )
    axis.legend(ncols=2)
    save_figure(figure, output_directory / f'{city.key}-throughput-comparison.png')


def plot_freiburg_validation(accumulator: EventAccumulator, output_directory: Path) -> None:
    city = CITIES[-1]
    metrics = (
        ('throughput_per_hour', 'Throughput', 'vehicles/hour'),
        ('completion_rate', 'Completion rate', 'fraction'),
        ('average_wait_density_s_per_m', 'Average wait density', 's/m'),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.2, 9.4), sharex=True)
    for axis, (metric, label, unit) in zip(axes, metrics, strict=True):
        learned = scalar_series(accumulator, evaluation_tag(city, 'learned', metric))
        axis.plot(
            learned.steps,
            learned.values,
            marker='o',
            markersize=3.0,
            linewidth=2.0,
            color='#276FBF',
            label='Learned (sampled)',
        )
        axis.axhline(
            baseline_value(accumulator, city, 'max-pressure', metric),
            color='#C44900',
            linewidth=1.7,
            label='Max pressure',
        )
        axis.axhline(
            baseline_value(accumulator, city, 'queue', metric),
            color='#4C956C',
            linewidth=1.7,
            linestyle='--',
            label='Queue',
        )
        mark_selected_iteration(axis)
        selected_index = learned.steps.index(SELECTED_ITERATION)
        axis.scatter(
            (SELECTED_ITERATION,), (learned.values[selected_index],), color='#222222', marker='D', s=38, zorder=5
        )
        axis.set_ylabel(f'{label}\n({unit})')
    axes[0].set_title('Freiburg validation trajectory across the full training run')
    axes[0].legend(ncols=3)
    axes[-1].set_xlabel('PPO iteration')
    save_figure(figure, output_directory / 'freiburg-validation-full-run.png')


def plot_diagnostics(accumulator: EventAccumulator, output_directory: Path) -> None:
    panels = (
        (('episode/mean_reward',), ('Mean reward',), 'Reward'),
        (('episode/mean_return',), ('Mean return',), 'Return'),
        (('diagnostics/explained_variance',), ('Explained variance',), 'Explained variance'),
        (('diagnostics/normalized_entropy',), ('Normalized entropy',), 'Entropy'),
        (('diagnostics/approximate_kl',), ('Approximate KL',), 'KL divergence'),
        (('train/policy_loss', 'train/value_loss'), ('Policy loss', 'Value loss'), 'Loss'),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True)
    for axis, (tags, labels, ylabel) in zip(axes.flat, panels, strict=True):
        for tag, label in zip(tags, labels, strict=True):
            series = scalar_series(accumulator, tag)
            axis.plot(series.steps, series.values, linewidth=1.6, label=label)
        mark_selected_iteration(axis)
        axis.set_ylabel(ylabel)
        if len(tags) > 1:
            axis.legend()
    figure.suptitle('PPO training diagnostics across the full run', fontweight='bold')
    for axis in axes[-1]:
        axis.set_xlabel('PPO iteration')
    figure.tight_layout()
    save_figure(figure, output_directory / 'ppo-training-diagnostics-full-run.png')


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    accumulator = load_events(arguments.event_dir)
    plot_city_throughput(accumulator, arguments.output_dir)
    for city in CITIES:
        plot_city_comparison(accumulator, arguments.output_dir, city)
    plot_freiburg_validation(accumulator, arguments.output_dir)
    plot_diagnostics(accumulator, arguments.output_dir)
    print(f'Wrote result figures to {arguments.output_dir}')


if __name__ == '__main__':
    main()
