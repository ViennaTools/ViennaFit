import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_domain():
    """Create a mock ViennaPS domain."""
    domain = Mock()
    domain.getLevelSets.return_value = [Mock()]
    return domain


@pytest.fixture
def mock_level_set():
    """Create a mock ViennaLS level set."""
    return Mock()


@pytest.fixture
def sample_project_info():
    """Sample project info dictionary."""
    return {
        "projectName": "TestProject",
        "projectDescription": "Test project description",
        "projectPath": "/test/path",
        "createdDate": "2024-01-01T00:00:00",
        "lastModifiedDate": "2024-01-01T00:00:00",
        "mode": "2D",
        "initialDomainPath": "",
        "targetLevelSetPath": "",
    }