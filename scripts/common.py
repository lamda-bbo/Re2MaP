import sys
import os
from functools import reduce

dirname = os.path.dirname
abspath = os.path.abspath
join = lambda *args: lambda former: os.path.join(former, *args)

relative_path = lambda path, *ops: reduce(lambda x, f: f(x), ops, path)

remap_root = relative_path(__file__, abspath, dirname, dirname)
dreamplace_root = relative_path(remap_root, join("dreamplace"))

sys.path.append(remap_root)
sys.path.append(dreamplace_root)

import logging

logging.root.name = 'ReMaP'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)