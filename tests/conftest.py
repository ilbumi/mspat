from pathlib import Path

import pytest
from padata.transform.base import BaseTransform
from padata.transform.bonds import AddResidueBonds
from padata.transform.compose import ComposeTransform
from padata.transform.protonation import ProtonateStructure


@pytest.fixture(scope="session", name="test_root")
def test_root_path() -> Path:
    """Test root path."""
    return Path(__file__).absolute().expanduser().resolve().parents[0]


@pytest.fixture
def structure_preprocessor() -> BaseTransform:
    """Preprocess protein structure."""
    return ComposeTransform([AddResidueBonds(), ProtonateStructure()])
