import pytest
from unittest.mock import Mock
from viennafit.fitOptimization import Optimization


def testOptimizationInit():
    """Test Optimization initialization."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    assert opt.name == "TestOpt"
    assert opt.studyType == "optimizationRuns"


def testSetVariableParameters():
    """Test setting variable parameters."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    opt.setParameterNames(["param1", "param2"])
    opt.setVariableParameters({"param1": (0.0, 10.0), "param2": (1.0, 5.0)})
    
    assert "param1" in opt.variableParameters
    assert opt.variableParameters["param1"] == (0.0, 10.0)


def testSetVariableParametersBeforeNames():
    """Test setting variable parameters before parameter names raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    with pytest.raises(ValueError, match="Parameter names must be set"):
        opt.setVariableParameters({"param1": (0.0, 10.0)})


def testSetVariableParametersInvalidBounds():
    """Test setting variable parameters with invalid bounds raises error."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    opt.setParameterNames(["param1"])
    with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
        opt.setVariableParameters({"param1": (10.0, 5.0)})


def testSetOptimizer():
    """Test setting optimizer."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    opt.setOptimizer("dlib")
    assert opt.optimizer == "dlib"


def testParameterConflictValidation():
    """Test parameter conflict validation."""
    mockProject = Mock()
    mockProject.isReady.return_value = True
    mockProject.projectPath = "/test/path"
    
    opt = Optimization("TestOpt", mockProject)
    opt.setParameterNames(["param1", "param2"])
    opt.setFixedParameters({"param1": 1.0})
    
    # Should raise error when trying to set same parameter as variable
    with pytest.raises(ValueError, match="conflicts with fixed"):
        opt.setVariableParameters({"param1": (0.0, 10.0)})