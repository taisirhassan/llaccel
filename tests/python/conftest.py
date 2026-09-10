from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO
