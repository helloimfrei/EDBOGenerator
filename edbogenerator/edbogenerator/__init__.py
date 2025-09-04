from .generator import EDBOGenerator

import sys
if sys.version_info >= (3, 10):
    raise RuntimeError("Python 3.10+ not supported with EDBO 0.2.0 / torch 1.10. "
                       "Use Python 3.8–3.9.")
