import pytest
import tempfile
from unittest.mock import Mock
from viennafit.fitStudy import Study
from viennafit.fitProject import Project


def testStudyInit():
    """Test Study initialization."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    assert study.name == "TestStudy"
    assert study.project == mockProject
    assert study.studyType == "testType"


def testStudyInitNotReadyProject():
    """Test Study initialization with unready project raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = False
    
    with pytest.raises(ValueError, match="Project is not ready"):
        Study("TestStudy", mockProject, "testType")


def testSetParameterNames():
    """Test setting parameter names."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    study.setParameterNames(["param1", "param2"])
    assert study.parameterNames == ["param1", "param2"]


def testSetFixedParameters():
    """Test setting fixed parameters."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    study.setParameterNames(["param1", "param2"])
    study.setFixedParameters({"param1": 1.0})
    assert study.fixedParameters == {"param1": 1.0}


def testSetFixedParametersBeforeNames():
    """Test setting fixed parameters before parameter names raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    with pytest.raises(ValueError, match="Parameter names must be set"):
        study.setFixedParameters({"param1": 1.0})


def testSetDistanceMetric():
    """Test setting distance metric."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    study.setDistanceMetric("CA+CSF")
    assert study.distanceMetric == "CA+CSF"


def testSetInvalidDistanceMetric():
    """Test setting invalid distance metric raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    study = Study("TestStudy", mockProject, "testType")
    with pytest.raises(ValueError, match="Invalid distance metric"):
        study.setDistanceMetric("INVALID")