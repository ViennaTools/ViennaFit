import pytest
from unittest.mock import Mock
from viennafit.fitGlobalSensitivityStudy import GlobalSensitivityStudy


def testGlobalSensitivityStudyInit():
    """Test GlobalSensitivityStudy initialization."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    assert gss.name == "TestGSS"
    assert gss.studyType == "globalSensStudies"


def testSetVariableParametersValidBounds():
    """Test setting variable parameters with valid bounds."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    gss.setParameterNames(["param1", "param2"])
    gss.setVariableParameters({"param1": (0.0, 10.0), "param2": (1.0, 5.0)})
    
    assert "param1" in gss.variableParameters
    assert gss.variableParameters["param1"] == (0.0, 10.0)


def testSetVariableParametersInvalidBounds():
    """Test setting variable parameters with invalid bounds raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    gss.setParameterNames(["param1"])
    
    with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
        gss.setVariableParameters({"param1": (10.0, 5.0)})


def testSetSamplingOptions():
    """Test setting sampling options."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    gss.setSamplingOptions(numSamples=100, secondOrder=True)
    
    assert gss.numSamples == 100
    assert gss.secondOrder == True


def testSetSamplingOptionsInvalidSamples():
    """Test setting sampling options with invalid sample count raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    
    with pytest.raises(ValueError, match="Number of samples must be positive"):
        gss.setSamplingOptions(numSamples=0)


def testSetSamplingMethod():
    """Test setting sampling method."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    gss.setSamplingMethod("fast")
    assert gss.samplingMethod == "fast"
    
    gss.setSamplingMethod("saltelli")
    assert gss.samplingMethod == "saltelli"


def testSetInvalidSamplingMethod():
    """Test setting invalid sampling method raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    gss = GlobalSensitivityStudy("TestGSS", mockProject)
    
    with pytest.raises(ValueError, match="Invalid sampling method"):
        gss.setSamplingMethod("invalid_method")