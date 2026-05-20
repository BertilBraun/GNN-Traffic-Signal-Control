"""Audit TL junction types in city.net.xml.

Prints a table of each traffic-light junction with its incoming-arm count
so we can decide which ones to keep and which to exclude.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo'), 'tools'))
import sumolib

HERE = Path(__file__).parent
NET = sumolib.net.readNet(str(HERE / 'city.net.xml'), withConnections=True)

counts = {3: [], 4: [], 'other': []}

for node in sorted(NET.getNodes(), key=lambda n: n.getID()):
    if node.getType() != 'traffic_light':
        continue
    n_in = len(node.getIncoming())
    if n_in == 3:
        counts[3].append(node.getID())
    elif n_in == 4:
        counts[4].append(node.getID())
    else:
        counts['other'].append((node.getID(), n_in))

print(f'3-way TL junctions ({len(counts[3])}):')
for jid in counts[3]:
    print(f'  {jid}')

print(f'\n4-way TL junctions ({len(counts[4])}):')
for jid in counts[4]:
    print(f'  {jid}')

print(f'\n5+-way TL junctions ({len(counts["other"])}) — WILL BE SKIPPED:')
for jid, n in counts['other']:
    print(f'  {jid}  ({n} arms)')

total_tl = len(counts[3]) + len(counts[4]) + len(counts['other'])
usable = len(counts[3]) + len(counts[4])
print(f'\nTotal TL junctions : {total_tl}')
print(f'Usable (3- or 4-way): {usable}')
print(f'Skipped (5+)        : {len(counts["other"])}')
