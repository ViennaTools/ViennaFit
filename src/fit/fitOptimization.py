from .fitProject import Project
from .fitProcessSequence import ProcessSequence
from viennaps2d import Domain
import importlib.util
import sys
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class Parameter:
    """Class to store parameter information"""
    name: str
    value: float = None
    minValue: float = None
    maxValue: float = None
    isFixed: bool = False

class Optimization:
    def __init__(self, name: str, project: Project):
        self.name = name
        self.project = project
        self.initialDomain = None
        self.resultLevelSet = None
        self.applied = False
        self.processSequence = None
        
        # Parameter handling
        self.parameters: Dict[str, Parameter] = {}
        self.bestParameters = None
        self.bestScore = float('inf')
        
        # Create directory for this optimization run, throw an error if it already exists
        self.runDir = os.path.join(project.projectPath, "optRuns", name)
        if os.path.exists(self.runDir):
            raise FileExistsError(f"Directory already exists: {self.runDir}. Please choose a different name.")
        os.makedirs(self.runDir)
        
    def addParameter(self, name: str, defaultValue: float = None):
        """
        Add a parameter that can be used in optimization
        
        Args:
            name: Parameter name
            defaultValue: Default value for the parameter (optional)
        """
        self.parameters[name] = Parameter(name=name, value=defaultValue)
        return self
        
    def addParameters(self, paramNames: List[str]):
        """
        Add multiple parameters that can be used in optimization
        
        Args:
            paramNames: List of parameter names
        """
        for name in paramNames:
            self.addParameter(name)
        return self
        
    def setFixedParameter(self, name: str, value: float):
        """
        Set a parameter as fixed with a specific value
        
        Args:
            name: Parameter name
            value: Fixed value for this parameter
        """
        if name not in self.parameters:
            self.addParameter(name)
            
        self.parameters[name].value = value
        self.parameters[name].isFixed = True
        return self
        
    def setFixedParameters(self, fixedParams: Dict[str, float]):
        """
        Set multiple parameters as fixed with specific values
        
        Args:
            fixedParams: Dictionary mapping parameter names to fixed values
        """
        for name, value in fixedParams.items():
            self.setFixedParameter(name, value)
        return self
        
    def setVariableParameter(self, name: str, minValue: float, maxValue: float):
        """
        Set a parameter as variable with a range for optimization
        
        Args:
            name: Parameter name
            minValue: Minimum value for optimization
            maxValue: Maximum value for optimization
        """
        if name not in self.parameters:
            self.addParameter(name)
            
        param = self.parameters[name]
        param.minValue = minValue
        param.maxValue = maxValue
        param.isFixed = False
        
        # Set to middle of range if no value exists
        if param.value is None:
            param.value = (minValue + maxValue) / 2
            
        return self
        
    def setVariableParameters(self, varParams: Dict[str, Tuple[float, float]]):
        """
        Set multiple parameters as variable with ranges
        
        Args:
            varParams: Dictionary mapping parameter names to tuples of (minValue, maxValue)
        """
        for name, (minVal, maxVal) in varParams.items():
            self.setVariableParameter(name, minVal, maxVal)
        return self
    
    def getParameterDict(self):
        """Get a dictionary of all parameter values"""
        return {name: param.value for name, param in self.parameters.items()}
    
    def getVariableParameterList(self):
        """Get list of variable parameters for optimization algorithms"""
        return [param for param in self.parameters.values() if not param.isFixed]
    
    def getVariableBounds(self):
        """Get bounds for variable parameters as lists"""
        varParams = self.getVariableParameterList()
        lowerBounds = [p.minValue for p in varParams]
        upperBounds = [p.maxValue for p in varParams]
        return lowerBounds, upperBounds
    
    def loadProcessSequence(self, filePath: str):
        """Load a process sequence from a Python file"""
        # Get absolute path
        absPath = os.path.abspath(filePath)
        
        if not os.path.exists(absPath):
            raise FileNotFoundError(f"Process sequence file not found: {absPath}")
            
        # Extract filename without extension
        moduleName = os.path.splitext(os.path.basename(absPath))[0]
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(moduleName, absPath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[moduleName] = module
        spec.loader.exec_module(module)
        
        # Try to find a ProcessSequence subclass in the module
        for itemName in dir(module):
            item = getattr(module, itemName)
            if isinstance(item, type) and issubclass(item, ProcessSequence) and item != ProcessSequence:
                # Create an instance of the found ProcessSequence subclass
                sequence = item()
                self.processSequence = sequence
                
                # Set the initial domain if it exists
                if self.initialDomain is not None:
                    sequence.setInitialDomain(self.initialDomain)
                    
                print(f"Successfully loaded process sequence: {itemName}")
                return self
                
        raise ValueError(f"No ProcessSequence subclass found in file: {absPath}")
    
    def setProcessSequence(self, sequence: ProcessSequence):
        """Set the process sequence to be optimized"""
        self.processSequence = sequence
        
        # Automatically set the initial domain to the process sequence
        if self.initialDomain is not None:
            self.processSequence.setInitialDomain(self.initialDomain)
            
        return self
    
    def setInitialDomain(self, passedDomain: Domain):
        """Set the initial domain for the optimization"""
        self.initialDomain = passedDomain
        
        # Also set the initial domain for the process sequence if it exists
        if self.processSequence is not None:
            self.processSequence.setInitialDomain(passedDomain)
        
        # Reset result since we have a new initial domain
        self.resultLevelSet = None
        self.applied = False
        return self
    
    def saveParameters(self, filename: str = "parameters.json"):
        """Save parameter configuration to file"""
        filepath = os.path.join(self.runDir, filename)
        
        # Convert parameters to serializable dictionary
        paramDict = {
            name: {
                "value": param.value,
                "minValue": param.minValue,
                "maxValue": param.maxValue,
                "isFixed": param.isFixed
            }
            for name, param in self.parameters.items()
        }
        
        with open(filepath, "w") as f:
            json.dump(paramDict, f, indent=4)
            
        print(f"Parameters saved to {filepath}")
        
    def saveBestResult(self, filename: str = "bestResult.json"):
        """Save best optimization result to file"""
        filepath = os.path.join(self.runDir, filename)
        
        result = {
            "bestScore": self.bestScore,
            "bestParameters": self.bestParameters
        }
        
        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"Best result saved to {filepath}")

    def objectiveFunction(self, x, paramNames=None):
        """
        Objective function for optimization
        
        Args:
            x: List of parameter values for variable parameters
            paramNames: Optional list of parameter names corresponding to x values
                      If None, uses the order from getVariableParameterList()
        
        Returns:
            Objective function value
        """
        # Get variable parameters in correct order
        varParams = self.getVariableParameterList()
        
        # If parameter names are provided, use them to map values
        if paramNames is None:
            paramNames = [p.name for p in varParams]
            
        # Update parameter values
        for i, value in enumerate(x):
            name = paramNames[i]
            self.parameters[name].value = value
            
        # Get complete parameter dictionary including fixed ones
        paramsDict = self.getParameterDict()
        
        # Convert to a simple object with attributes for easier access in the process sequence
        class Params:
            pass
        
        paramsObj = Params()
        for name, value in paramsDict.items():
            setattr(paramsObj, name, value)
            
        # Apply process sequence with current parameters
        self.processSequence.apply(paramsObj)
        
        # Get result (this would be based on a comparison with target data)
        # For now, this is just a placeholder
        result = 0.0  # This should be calculated based on your objective function
        
        # Update best result if this is better
        if result < self.bestScore:
            self.bestScore = result
            self.bestParameters = paramsDict.copy()
            
        return result

    def optimize(self):
        """Run the optimization"""
        # Save initial parameter configuration
        self.saveParameters("initialParameters.json")
        
        print("Running optimization...")
        
        # After optimization, save results
        self.saveBestResult()
        self.saveParameters("finalParameters.json")
