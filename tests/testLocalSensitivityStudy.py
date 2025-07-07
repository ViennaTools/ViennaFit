import pytest
from unittest.mock import Mock
from viennafit.fitLocalSensitivityStudy import LocalSensitivityStudy


def testLocalSensitivityStudyInit():
    """Test LocalSensitivityStudy initialization."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    lss = LocalSensitivityStudy("TestLSS", mockProject)
    assert lss.name == "TestLSS"
    assert lss.studyType == "locSensStudies"


def testSetVariableParametersValidBounds():
    """Test setting variable parameters with valid bounds."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    lss = LocalSensitivityStudy("TestLSS", mockProject)
    lss.setParameterNames(["param1", "param2"])
    lss.setVariableParameters(
        {"param1": (0.0, 5.0, 10.0), "param2": (1.0, 3.0, 5.0)}, 
        (5, 7)
    )
    
    assert "param1" in lss.variableParameters
    assert lss.variableParameters["param1"] == (0.0, 5.0, 10.0)
    assert lss.nEval == (5, 7)


def testSetVariableParametersInvalidPOI():
    """Test setting variable parameters with POI outside bounds raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    lss = LocalSensitivityStudy("TestLSS", mockProject)
    lss.setParameterNames(["param1"])
    
    # POI below lower bound
    with pytest.raises(ValueError, match="Point of interest.*must be within bounds"):
        lss.setVariableParameters({"param1": (5.0, 3.0, 10.0)}, (5,))
    
    # POI above upper bound  
    with pytest.raises(ValueError, match="Point of interest.*must be within bounds"):
        lss.setVariableParameters({"param1": (0.0, 15.0, 10.0)}, (5,))


def testSetVariableParametersWrongNEvalCount():
    """Test setting variable parameters with wrong nEval count raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    lss = LocalSensitivityStudy("TestLSS", mockProject)
    lss.setParameterNames(["param1", "param2"])
    
    with pytest.raises(ValueError, match="nEval tuple must have same length"):
        lss.setVariableParameters(
            {"param1": (0.0, 5.0, 10.0), "param2": (1.0, 3.0, 5.0)}, 
            (5,)  # Only one value for two parameters
        )