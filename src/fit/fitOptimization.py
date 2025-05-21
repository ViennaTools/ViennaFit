from .fitProject import Project
from .fitProcessSequence import ProcessSequence
from .fitDistanceMetrics import constuctDistanceMetric
from viennaps2d import Domain
import importlib.util
import sys
import os
import json
from typing import Dict, List, Tuple, Any, Optional


class Optimization:
    def __init__(self, name: str, project: Project):
        self.name = name

        # check project readiness, and set run directory
        if project.isReady:
            self.runDir = os.path.join(project.projectPath, "optimizationRuns", name)
        else:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains, and then run the optimization."
            )
        os.makedirs(self.runDir, exist_ok=True)
        self.project = project

        # Set internal variables
        self.resultLevelSet = None
        self.applied = False
        self.processSequence = None
        self.distanceMetric = None

        # Parameter handling
        self.parameterNames = []
        self.fixedParameters = {}
        self.variableParameters = {}
        self.bestParameters = None
        self.bestScore = float("inf")

        print(
            f"Optimization '{self.name}' assigned to project '{self.project.projectName}' and initialized in {self.runDir}"
        )

    def setParameterNames(self, paramNames: List[str]):
        """Specifies names of parameters that will be used in optimization"""
        self.parameterNames = paramNames
        return self

    def setFixedParameters(self, fixedParams: Dict[str, float]):
        """
        Set multiple parameters as fixed with specific values

        Args:
            fixedParams: Dictionary mapping parameter names to fixed values
        """
        if self.parameterNames is None:
            raise ValueError(
                "Parameter names must be set before defining fixed parameters"
            )
        for name, value in fixedParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names"
                )
            if name in self.variableParameters:
                raise ValueError(f"Parameter '{name}' is already set as variable")
            self.fixedParameters[name] = value
        return self

    def setVariableParameters(self, varParams: Dict[str, Tuple[float, float]]):
        """
        Set multiple parameters as variable with ranges

        Args:
            varParams: Dictionary mapping parameter names to tuples of (lowerBound, upperBound)
        """
        if self.parameterNames is None:
            raise ValueError(
                "Parameter names must be set before defining variable parameters"
            )
        for name, (lowerBound, upperBound) in varParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names. \
                    The current parameter names are: {self.parameterNames}"
                )
            if name in self.fixedParameters:
                raise ValueError(f"Parameter '{name}' is already set as fixed")
            self.variableParameters[name] = (lowerBound, upperBound)
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
        lowerBounds = [p.lowerBound for p in varParams]
        upperBounds = [p.upperBound for p in varParams]
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
            if (
                isinstance(item, type)
                and issubclass(item, ProcessSequence)
                and item != ProcessSequence
            ):
                # Create an instance of the found ProcessSequence subclass
                sequence = item()
                self.processSequence = sequence

                # Set the initial domain if it exists
                if self.initialDomain is not None:
                    sequence.setInitialDomain(self.initialDomain)

                print(f"Successfully loaded process sequence: {itemName}")
                return self

        raise ValueError(f"No ProcessSequence subclass found in file: {absPath}")

    def setProcessSequence(self, sequence_func):
        """
        Set the process sequence to be optimized.

        Args:
            sequence_func: Function with signature (domain, *, param1, param2, ...) -> domain
                The function should accept an initial domain and parameter keywords,
                apply the process sequence, and return the resulting domain.

        Example:
            def my_sequence(domain, *, param1, param2):
                model = vps.MultiParticleProcess()
                # Configure process with params
                # ...
                return processed_domain

            opt.setProcessSequence(my_sequence)
        """
        # Validate the function signature
        import inspect

        sig = inspect.signature(sequence_func)

        # Check if first parameter exists and is positional
        params = list(sig.parameters.values())
        if not params or params[0].kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise ValueError("Process sequence must accept domain as first argument")

        self.processSequence = sequence_func

        # Store parameter names from function signature
        self.sequence_params = set()
        for param in params[1:]:  # Skip first param (domain)
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                self.sequence_params.add(param.name)

        print(
            f"Process sequence set with parameters: {', '.join(self.sequence_params)}"
        )
        return self

    def saveParameters(self, filename: str = "parameters.json"):
        """Save parameter configuration to file"""
        filepath = os.path.join(self.runDir, filename)

        # Convert parameters to serializable dictionary
        paramDict = {
            name: {
                "value": param.value,
                "lowerBound": param.lowerBound,
                "upperBound": param.upperBound,
                "isFixed": param.isFixed,
            }
            for name, param in self.parameters.items()
        }

        with open(filepath, "w") as f:
            json.dump(paramDict, f, indent=4)

        print(f"Parameters saved to {filepath}")

    def saveBestResult(self, filename: str = "bestResult.json"):
        """Save best optimization result to file"""
        filepath = os.path.join(self.runDir, filename)

        result = {"bestScore": self.bestScore, "bestParameters": self.bestParameters}

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

    def validate(self):
        """Validate that all required parameters are defined"""
        if not hasattr(self, "processSequence"):
            raise ValueError("No process sequence has been set")

        if not self.parameterNames:
            raise ValueError("No parameters have been defined")

        # if the union of the fixed and variable parameters is not equal to the parameter names
        if set(self.fixedParameters.keys()).union(
            set(self.variableParameters.keys())
        ) != set(self.parameterNames):
            raise ValueError(
                "The union of fixed and variable parameters does not match the parameter names"
            )

        return True

    def setDistanceMetric(self, metric: str):
        """
        Set the distance metric for comparing level sets which will be minimized

        Args:
            metric: Distance metric to be used.
                    Options include:
                        - 'CA' (Compare area mismatch),
                        - 'CSF' (Compare sparse field),
                        - 'CNB' (Compare narrow band),
                        - 'CA+CSF' (Sum of CA and CSF),
                        - 'CA+CNB' (Sum of CA and CNB).
        """

        if metric not in ["CA", "CSF", "CNB", "CA+CSF", "CA+CNB"]:
            raise ValueError(
                f"Invalid distance metric: {metric}. "
                "Options are 'CA', 'CSF', 'CNB', 'CA+CSF', 'CA+CNB'."
            )

        pass  # Placeholder for actual distance metric implementation
        return self
