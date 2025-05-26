from pathlib import Path

import pytest


@pytest.fixture(scope="session", name="test_root")
def test_root_path() -> Path:
    """Test root path."""
    return Path(__file__).absolute().expanduser().resolve().parents[0]
