"""Pytest bootstrap for the Developer-2 test suite.

Sets a throwaway ``CD4_DATA_ROOT`` **before** ``core.contract`` is imported, so tests
that write ``runs/``, ``checkpoints/``, etc. land in a temp dir, never the real one.
Adds the repo root to sys.path so ``import core...`` works from anywhere.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set before the first `import core.contract` (contract reads it at import).
os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-test-data-"))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_dirs():
    from core import contract

    contract.ensure_dirs()
    yield
