from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.movement.evaluation.metrics import parse_tripinfo_metrics


def test_parse_tripinfo_metrics_includes_time_loss(tmp_path: Path) -> None:
    tripinfo_path = tmp_path / 'tripinfo.xml'
    tripinfo_path.write_text(
        '<tripinfos>'
        '<tripinfo waitingTime="10" duration="100" timeLoss="25"/>'
        '<tripinfo waitingTime="20" duration="140" timeLoss="35"/>'
        '</tripinfos>',
        encoding='utf-8',
    )

    completed, throughput, waiting_time, travel_time, time_loss = parse_tripinfo_metrics(
        tripinfo_path=tripinfo_path,
        episode_length_s=600,
    )

    assert completed == 2
    assert throughput == 12.0
    assert waiting_time == 15.0
    assert travel_time == 120.0
    assert time_loss == 30.0
