from .fitProject import Project
from .fitProcessSequence import ProcessSequence
from .fitOptimizerWrapper import OptimizerWrapper
import viennaps2d as vps
import importlib.util
import sys
import os
import json
import inspect
from typing import Dict, List, Tuple


class Optimization:
    def __init__(self, name: str, project: Project):
        self.name = name

        # check project readiness, and set run directory
        if project.isReady:
            self.runDir = os.path.join(project.projectPath, "optimizationRuns", name)
            self.progressDir = os.path.join(self.runDir, "progress")
        else:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains, and then run the optimization."
            )
        if os.path.exists(self.runDir):
            raise FileExistsError(
                f"Run directory already exists: {self.runDir}. \n "
                "Please choose a different name or delete the existing directory."
            )
        os.makedirs(self.runDir, exist_ok=False)
        os.makedirs(self.progressDir, exist_ok=False)
        self.project = project

        # Set internal variables
        self.resultLevelSet = None
        self.applied = False
        self.processSequence = None
        self.distanceMetric = None
        self.optimizer = "dlib"

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
        return self.variableParameters.keys()

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

        # Look for function with correct signature
        for itemName in dir(module):
            item = getattr(module, itemName)
            if callable(item) and not isinstance(item, type):
                try:
                    sig = inspect.signature(item)
                    params = list(sig.parameters.values())

                    # Check function has exactly 2 parameters
                    if len(params) == 2:
                        # Check parameter types
                        if (
                            params[0].annotation == vps.Domain
                            or params[0].annotation == inspect.Parameter.empty
                        ):
                            if (
                                params[1].annotation == dict[str, float]
                                or params[1].annotation == inspect.Parameter.empty
                            ):
                                self.processSequence = item
                                print(
                                    f"Successfully loaded process sequence: {itemName}"
                                )
                                return self
                except ValueError:
                    continue

        raise ValueError(
            f"No suitable process sequence function found in file: {absPath}"
        )

    def setProcessSequence(self, processFunction):
        """
        Set the process sequence to be optimized.

        Args:
            processFunction: Function with signature:
                (domain: viennaps2d.Domain, params: dict[str, float]) -> viennaps2d.Domain
                The function should take an initial domain and parameter dictionary,
                apply the process sequence, and return the resulting domain.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If function signature doesn't match requirements
            TypeError: If parameter types don't match requirements
        """
        functionSignature = inspect.signature(processFunction)
        functionParams = list(functionSignature.parameters.values())

        # Check function has exactly 2 parameters
        if len(functionParams) != 2:
            raise ValueError(
                f"Process sequence must have exactly 2 parameters, got {len(functionParams)}"
            )

        # Check first parameter (domain)
        domainParam = functionParams[0]
        if (
            domainParam.annotation != vps.Domain
            and domainParam.annotation != inspect.Parameter.empty
        ):
            raise TypeError(
                f"First parameter must be a viennaps2d.Domain, got {domainParam.annotation}"
            )

        # Check second parameter (params dict)
        paramsParam = functionParams[1]
        expectedType = dict[str, float]
        if (
            paramsParam.annotation != expectedType
            and paramsParam.annotation != inspect.Parameter.empty
        ):
            raise TypeError(
                f"Second parameter must be Dict[str, float], got {paramsParam.annotation}"
            )

        self.processSequence = processFunction

        # Save passed process function into a file
        processFilePath = os.path.join(self.runDir, f"{self.name}-processSequence.py")

        # Get the source code of the function
        source = inspect.getsource(processFunction)

        # Save to file with imports at top
        with open(processFilePath, "w") as f:
            f.write("import viennaps2d as vps\n")
            f.write("import viennals2d as vls\n\n")
            f.write(source)

        print(f"Process sequence set: {processFunction.__name__}")
        print(f"Process sequence saved to: {processFilePath}")
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

    def validate(self):
        """Validate that all required parameters are defined"""
        if not hasattr(self, "processSequence"):
            raise ValueError("No process sequence has been set")

        if not self.parameterNames:
            raise ValueError("No parameters have been defined")

        if not self.distanceMetric:
            raise ValueError("No distance metric has been set")

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

        self.distanceMetric = metric
        return self

    def setOptimizer(self, optimizer: str):
        """
        Set the optimizer to be used for the optimization process.

        Args:
            optimizer: The name of the optimizer to be used.
        """
        self.optimizer = optimizer
        return self

    def apply(
        self,
        numEvaluations: int = 100,
        saveAllEvaluations: bool = False,
        saveVisualization: bool = True,
    ):
        """Apply the optimization."""
        if not self.applied:
            self.validate()
            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations
            self.applied = True
            self.evalCounter = 0
        else:
            print("Optimization has already been applied.")

        # Create optimizer wrapper
        optimizer = OptimizerWrapper.create(self.optimizer, self)

        try:
            # Run optimization
            result = optimizer.optimize(numEvaluations)

            # Save results
            if result["success"]:
                self.bestParameters.update(result["x"])
                self.bestScore = result["fun"]

                print(f"Optimization completed successfully:")
                print(f"  Function evaluations: {result['nfev']}")
                print(f"  Best score: {result['fun']:.6f}")
                print("  Best parameters:")
                for name, value in result["x"].items():
                    print(f"    {name}: {value:.6f}")

                # Save final results
                self.saveBestResult()

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise
