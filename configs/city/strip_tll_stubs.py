"""Remove empty tlLogic stubs that netconvert writes into city.net.xml.

netconvert --tls.guess-signals emits skeleton <tlLogic> elements with no
<phase> children.  Our city.tll.xml additional file supplies the real phases.
Having both causes SUMO to raise 'Another logic ... already exists'.

This script removes all <tlLogic> elements that have no <phase> children
from city.net.xml so only our additional file's programs are used.
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
NET_FILE = HERE / 'city.net.xml'

text = NET_FILE.read_text(encoding='utf-8')

# Match <tlLogic ...></tlLogic> blocks (possibly with whitespace) that have no
# nested <phase> element.  The pattern handles single-line and multi-line stubs.
pattern = re.compile(
    r'\s*<tlLogic\b[^>]*>\s*</tlLogic>',
    re.DOTALL,
)

cleaned, n = pattern.subn('', text)
NET_FILE.write_text(cleaned, encoding='utf-8')
print(f'Removed {n} empty tlLogic stubs from {NET_FILE}')
