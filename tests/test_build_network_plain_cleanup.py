from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_network import (
    NetconvertOsmImportOptions,
    _is_netconvert_internal_abort,
    _netconvert_osm_import_command,
    _strip_plain_edge_types,
)


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


def test_netconvert_osm_import_command_can_disable_sumo_import_helpers(tmp_path: Path) -> None:
    osm_path = tmp_path / 'city.osm'
    net_path = tmp_path / 'city.net.xml'

    full_command = _netconvert_osm_import_command(
        osm_path=osm_path,
        net_path=net_path,
        join_dist=40.0,
        netconvert='netconvert',
        import_options=NetconvertOsmImportOptions(
            label='full',
            join_junctions=True,
            guess_signal_clusters=True,
            remove_geometry_nodes=True,
        ),
    )
    fallback_command = _netconvert_osm_import_command(
        osm_path=osm_path,
        net_path=net_path,
        join_dist=40.0,
        netconvert='netconvert',
        import_options=NetconvertOsmImportOptions(
            label='minimal',
            join_junctions=False,
            guess_signal_clusters=False,
            remove_geometry_nodes=False,
        ),
    )

    assert '--junctions.join' in full_command
    assert '--tls.join' in full_command
    assert '--tls.guess-signals' in full_command
    assert '--geometry.remove' in full_command
    assert '--junctions.join' not in fallback_command
    assert '--tls.join' not in fallback_command
    assert '--tls.guess-signals' not in fallback_command
    assert '--geometry.remove' not in fallback_command
    assert fallback_command[:5] == ['netconvert', '--osm-files', str(osm_path), '--output-file', str(net_path)]


def test_is_netconvert_internal_abort_detects_assertion_and_signal_exit() -> None:
    assertion_result = subprocess.CompletedProcess(
        args=['netconvert'],
        returncode=1,
        stderr="netconvert: NBAlgorithms.h:193: Assertion `angle >= 0' failed.",
    )
    signal_result = subprocess.CompletedProcess(args=['netconvert'], returncode=-6, stderr='')
    input_error_result = subprocess.CompletedProcess(
        args=['netconvert'], returncode=1, stderr='Error: No edges loaded.'
    )

    assert _is_netconvert_internal_abort(assertion_result)
    assert _is_netconvert_internal_abort(signal_result)
    assert not _is_netconvert_internal_abort(input_error_result)
