import pytest
import tempfile
import os
from viennafit.fitProject import Project


def testProjectInit():
    """Test basic project initialization."""
    project = Project()
    assert project.projectName is None
    assert project.mode == "2D"


def testProjectInitWithName():
    """Test project initialization with name."""
    with tempfile.TemporaryDirectory() as tempDir:
        project = Project("TestProject", tempDir)
        assert project.projectName == "TestProject"
        assert "TestProject" in project.projectPath


def testSetMode():
    """Test setting project mode."""
    project = Project()
    project.setMode("3D")
    assert project.mode == "3D"
    
    with pytest.raises(ValueError):
        project.setMode("invalid")


def testInitializeCreatesStructure():
    """Test project initialization creates directory structure."""
    with tempfile.TemporaryDirectory() as tempDir:
        project = Project("TestProject", tempDir)
        project.initialize()
        
        assert os.path.exists(project.projectPath)
        assert os.path.exists(os.path.join(project.projectPath, "domains"))
        assert os.path.exists(project.projectInfoPath)


def testIsReady():
    """Test project readiness check."""
    project = Project()
    assert not project.isReady()


def testFileLoadingValidation():
    """Test file loading methods validate file existence and format."""
    project = Project()
    
    with pytest.raises(FileNotFoundError):
        project.setInitialDomainFromFile("/nonexistent.vpsd")
    
    with pytest.raises(FileNotFoundError):
        project.setTargetDomainFromFile("/nonexistent.lvst")
    
    # Test wrong file extensions
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        with pytest.raises(ValueError):
            project.setInitialDomainFromFile(f.name)
        with pytest.raises(ValueError):
            project.setTargetDomainFromFile(f.name)