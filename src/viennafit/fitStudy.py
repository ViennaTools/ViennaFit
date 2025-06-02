from .fitProject import Project
import viennaps2d as vps
import importlib.util
import sys
import os
import json
import inspect
from typing import Dict, List, Callable


class Study:
    """Base class for all parameter studies (optimization, sensitivity analysis, etc.)"""

    def __init__(self, name: str, project: Project, studyType: str):
        # self.name = name
        base_name = name
        self.project = project

        # Check project readiness
        if not project.isReady:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains before running the study."
            )

        # Set up directories
        run_dir = os.path.join(project.projectPath, studyType, base_name)
        if os.path.exists(run_dir):
            index = 1
            while True:
                alt_name = f"{base_name}_{index}"
                alt_run_dir = os.path.join(project.projectPath, studyType, alt_name)
                if not os.path.exists(alt_run_dir):
                    run_dir = alt_run_dir
                    name = alt_name
                    print(
                        f"Run directory already exists. Renaming study to '{name}' "
                        f"and using directory: {run_dir}"
                    )
                    break
                index += 1
            # raise FileExistsError(
            #     f"Run directory already exists: {self.runDir}. \n"
            #     "Please choose a different name or delete the existing directory."
            # )
        self.name = name
        self.runDir = run_dir
        os.makedirs(self.runDir, exist_ok=False)

        # Create progress directory (name may vary in subclasses)
        self.progressDir = os.path.join(self.runDir, "progress")
        os.makedirs(self.progressDir, exist_ok=False)

        # Common internal variables
        self.resultLevelSet = None
        self.applied = False
        self.processSequence = None
        self.distanceMetric = None
        self.evalCounter = 0
        self.saveVisualization = True
        self.saveAllEvaluations = False

        # Parameter handling
        self.parameterNames = []
        self.fixedParameters = {}
        self.variableParameters = {}
        self.bestParameters = None
        self.bestScore = float("inf")

        print(
            f"{studyType.capitalize()} '{self.name}' assigned to project '{self.project.projectName}'"
            f" and initialized in {self.runDir}"
        )

    def setParameterNames(self, paramNames: List[str]):
        """Specifies names of parameters that will be used in the study"""
        self.parameterNames = paramNames
        return self

    def setFixedParameters(self, fixedParams: Dict[str, float]):
        """
        Set multiple parameters as fixed with specific values

        Args:
            fixedParams: Dictionary mapping parameter names to fixed values
        """
        if not self.parameterNames:
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

    def setProcessSequence(self, processFunction: Callable):
        """
        Set the process sequence to be used in the study.

        Args:
            processFunction: Function with signature:
                (domain: viennaps2d.Domain, params: dict[str, float]) -> viennaps2d.Domain
                The function should take an initial domain and parameter dictionary,
                apply the process sequence, and return the resulting domain.

        Returns:
            self: For method chaining
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

    def saveBestResult(self, filename: str = "bestResult.json"):
        """Save best result to file"""
        filepath = os.path.join(self.runDir, filename)

        result = {"bestScore": self.bestScore, "bestParameters": self.bestParameters}

        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Best result saved to {filepath}")

    def validate(self):
        """Validate that all required parameters are defined"""
        if not hasattr(self, "processSequence") or self.processSequence is None:
            raise ValueError("No process sequence has been set")

        if not self.parameterNames:
            raise ValueError("No parameters have been defined")

        if not self.distanceMetric:
            raise ValueError("No distance metric has been set")

        if set(self.fixedParameters.keys()).union(
            set(self.variableParameters.keys())
        ) != set(self.parameterNames):
            raise ValueError(
                "The union of fixed and variable parameters does not match the parameter names"
            )

        return True

    def setDistanceMetric(self, metric: str):
        """
        Set the distance metric for comparing level sets.

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

    def apply(self, *args, **kwargs):
        """Apply the study - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement apply()")
