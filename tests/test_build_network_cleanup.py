from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_network import _remove_roundabouts_in_plain


def test_remove_roundabouts_in_plain_removes_stale_roundabout_metadata(tmp_path: Path) -> None:
    edg_file = tmp_path / "plain.edg.xml"
    edg_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="a" from="n0" to="n1"/>
    <roundabout nodes="n0 n1" edges="a missing"/>
</edges>
""",
        encoding="utf-8",
    )

    removed = _remove_roundabouts_in_plain(edg_file)

    root = ET.parse(edg_file).getroot()
    assert removed == 1
    assert root.findall("roundabout") == []
    assert root.find("edge").get("id") == "a"
