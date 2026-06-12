from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_il_data import resolve_sumocfg_net_path


def test_resolve_sumocfg_net_path_uses_config_directory(tmp_path: Path) -> None:
    cfg = tmp_path / "nested" / "grid.sumocfg"
    cfg.parent.mkdir()
    cfg.write_text(
        """<configuration><input><net-file value="grid.net.xml"/></input></configuration>""",
        encoding="utf-8",
    )

    assert resolve_sumocfg_net_path(cfg) == cfg.parent / "grid.net.xml"
