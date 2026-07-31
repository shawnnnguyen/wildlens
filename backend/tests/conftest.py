"""
Single sys.path bootstrap for the whole backend/tests/ directory — pytest
loads this before collecting any test module in this directory, so
individual test files no longer need their own sys.path.insert boilerplate.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR   = Path(__file__).parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT   = _BACKEND_DIR.parent

sys.path.insert(0, str(_BACKEND_DIR))   # for `import wildlens`
sys.path.insert(0, str(_REPO_ROOT))     # for `import backend.api...`
