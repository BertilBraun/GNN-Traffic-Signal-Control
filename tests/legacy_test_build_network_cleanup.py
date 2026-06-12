from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_network import _remove_roundabouts_in_plain, _unsupported_tl_ids_by_incoming_count


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


class _FakeNode:
    def __init__(self, node_id: str, node_type: str, incoming_count: int) -> None:
        self._id = node_id
        self._type = node_type
        self._incoming = [object() for _ in range(incoming_count)]

    def getID(self) -> str:
        return self._id

    def getType(self) -> str:
        return self._type

    def getIncoming(self) -> list:
        return self._incoming


class _FakeNet:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self._nodes = nodes

    def getNodes(self) -> list[_FakeNode]:
        return self._nodes


def test_unsupported_tl_ids_by_incoming_count_selects_only_unrepresentable_tls() -> None:
    net = _FakeNet(
        [
            _FakeNode("two_arm", "traffic_light", 2),
            _FakeNode("three_arm", "traffic_light", 3),
            _FakeNode("four_arm", "traffic_light", 4),
            _FakeNode("priority_two_arm", "priority", 2),
            _FakeNode("five_arm", "traffic_light", 5),
        ]
    )

    assert _unsupported_tl_ids_by_incoming_count(net) == ["five_arm", "two_arm"]
