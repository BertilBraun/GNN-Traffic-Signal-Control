from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_network import _strip_plain_edge_types


def test_strip_plain_edge_types_removes_osm_type_references(tmp_path: Path) -> None:
    edg_file = tmp_path / 'plain.edg.xml'
    edg_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="a" from="n0" to="n1" priority="3" type="highway.residential" speed="8.33"/>
    <edge id="b" from="n1" to="n2" priority="10" speed="13.89"/>
</edges>
""",
        encoding='utf-8',
    )

    stripped = _strip_plain_edge_types(edg_file)

    edges_by_id = {str(edge.get('id')): edge for edge in ET.parse(edg_file).getroot().findall('edge')}
    assert stripped == 1
    assert 'type' not in edges_by_id['a'].attrib
    assert edges_by_id['a'].get('priority') == '3'
    assert edges_by_id['a'].get('speed') == '8.33'
    assert edges_by_id['b'].get('priority') == '10'
