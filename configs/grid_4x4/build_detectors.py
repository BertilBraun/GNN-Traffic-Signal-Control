"""Generate E2 detector .add.xml for all incoming lanes at TL junctions.
Detector length clipped to min(200m, lane_length) per PLAN §9.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

HERE = Path(__file__).parent
NET = sumolib.net.readNet(str(HERE / 'grid.net.xml'))
NOMINAL = 200.0  # nominal detector length (m)

root = ET.Element('additional')
for node in sorted(NET.getNodes(), key=lambda n: n.getID()):
    if node.getType() != 'traffic_light':
        continue
    for edge in node.getIncoming():
        for lane in edge.getLanes():
            lane_len = lane.getLength()
            det_len = min(NOMINAL, lane_len)
            pos = max(0.0, lane_len - det_len)
            ET.SubElement(
                root,
                'laneAreaDetector',
                id=f'det_{lane.getID()}',
                lane=lane.getID(),
                pos=f'{pos:.2f}',
                length=f'{det_len:.2f}',
                freq='3600',
                file='detector_out.xml',
                friendlyPos='true',
            )

xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
xml_str = '\n'.join(xml_str.split('\n')[1:])
out = HERE / 'grid.add.xml'
out.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str)
n = len(root)
print(f'Wrote {n} detectors -> {out}')
