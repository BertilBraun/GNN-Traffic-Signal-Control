"""Download OSM data for Munich Maxvorstadt via the Overpass API.

Bounding box (approx 900m x 700m):
  min_lat=48.147, min_lon=11.568, max_lat=48.155, max_lon=11.581

Expected ~15-25 signalised junctions, mostly 4-way.
"""

import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

BBOX = '48.147,11.568,48.155,11.581'  # south,west,north,east

QUERY = f"""
[out:xml][timeout:60];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street)$"]({BBOX});
  node(w)({BBOX});
  relation["type"="restriction"]({BBOX});
);
out body;
>;
out skel qt;
"""

URL = 'https://overpass-api.de/api/interpreter'

OUT = HERE / 'city.osm'


def main() -> None:
    print('Querying Overpass API ...')
    payload = urllib.parse.urlencode({'data': QUERY.strip()}).encode('utf-8')
    req = urllib.request.Request(URL, data=payload, method='POST')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    req.add_header('User-Agent', 'SUMO-GNN-research/1.0')
    with urllib.request.urlopen(req, timeout=90) as resp:
        content = resp.read()
    OUT.write_bytes(content)
    size_kb = OUT.stat().st_size // 1024
    print(f'Wrote {OUT}  ({size_kb} kB)')


if __name__ == '__main__':
    main()
