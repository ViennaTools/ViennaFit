from .fitProject import Project
import viennaps2d as vps
import importlib.util
import sys
import os
import json
import inspect
from typing import Callable


class Study:
    """Base class for all parameter studies (optimization, sensitivity analysis, etc.)"""

    def _generateRunDirectory(self, baseName: str, studyType: str):
        """Generate run directory with conflict resolution"""
        runDir = os.path.join(self.project.projectPath, studyType, baseName)
        name = baseName
        
        if os.path.exists(runDir):
            # Find the highest existing index
            studyDir = os.path.join(self.project.projectPath, studyType)
            maxIndex = 0

            if os.path.exists(studyDir):
                for existingDir in os.listdir(studyDir):
                    if (
                        existingDir.startswith(f"{baseName}_")
                        and existingDir[len(baseName) + 1 :].isdigit()
                    ):
                        existingIndex = int(existingDir[len(baseName) + 1 :])
                        maxIndex = max(maxIndex, existingIndex)

            # Use the next available index
            newIndex = maxIndex + 1
            name = f"{baseName}_{newIndex}"
            runDir = os.path.join(self.project.projectPath, studyType, name)
            print(
                f"Run directory already exists. Renaming study to '{name}' "
                f"and using directory: {runDir}"
            )
        
        return name, runDir

    def __init__(self, name: str, project: Project, studyType: str):
        # self.name = name
        baseName = name
        self.project = project

        # Check project readiness
        if not project.isReady:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains before running the study."
            )

        # Set up directory paths (but don't create them yet)
        name, runDir = self._generateRunDirectory(baseName, studyType)

        self.name = name
        self.runDir = runDir
        self.progressDir = os.path.join(self.runDir, "progress")

        # Common internal variables
        self.resultLevelSet = None
        self.applied = False
        self.processSequence = None
        self.distanceMetric = None
        self.criticalDimensionRanges = None
        self.evalCounter = 0
        self.saveVisualization = True
        self.saveAllEvaluations = False

        # Parameter handling
        self.parameterNames = []
        self.fixedParameters = {}
        self.variableParameters = {}
        self.bestParameters = None
        self.bestScore = float("inf")
        self.bestEvaluationNumber = None

        print(
            f"{studyType.capitalize()} '{self.name}' assigned to project '{self.project.projectName}'"
            f" and initialized in {self.runDir}"
        )

    def setParameterNames(self, paramNames: list[str]):
        """Specifies names of parameters that will be used in the study"""
        self.parameterNames = paramNames
        return self

    def setFixedParameters(self, fixedParams: dict[str, float]):
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
                        # Check parameter types - support both single and multi-domain
                        validDomainTypes = [
                            vps.Domain,  # Single domain (backward compatibility)
                            dict[str, vps.Domain],  # Multi-domain dict
                            list[vps.Domain],  # Multi-domain list
                            inspect.Parameter.empty  # No annotation
                        ]
                        
                        if params[0].annotation in validDomainTypes:
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
            processFunction: Function with one of these signatures:
                Single-domain (backward compatibility):
                    (domain: viennaps2d.Domain, params: dict[str, float]) -> viennaps2d.Domain
                Multi-domain:
                    (domains: dict[str, viennaps2d.Domain], params: dict[str, float]) -> dict[str, viennaps2d.Domain]
                The function should take initial domain(s) and parameter dictionary,
                apply the process sequence, and return the resulting domain(s).

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

        # Check first parameter (domain or domains)
        domainParam = functionParams[0]
        validDomainTypes = [
            vps.Domain,  # Single domain (backward compatibility)
            dict[str, vps.Domain],  # Multi-domain dict
            list[vps.Domain],  # Multi-domain list
            inspect.Parameter.empty  # No annotation
        ]
        
        if domainParam.annotation not in validDomainTypes:
            raise TypeError(
                f"First parameter must be viennaps2d.Domain, dict[str, viennaps2d.Domain], or list[viennaps2d.Domain], got {domainParam.annotation}"
            )

        # Check second parameter (params dict)
        paramsParam = functionParams[1]
        expectedType = dict[str, float]
        if (
            paramsParam.annotation != expectedType
            and paramsParam.annotation != inspect.Parameter.empty
        ):
            raise TypeError(
                f"Second parameter must be dict[str, float], got {paramsParam.annotation}"
            )

        self.processSequence = processFunction
        print(f"Process sequence set: {processFunction.__name__}")
        return self

    def _saveProcessSequence(self):
        """Save the process sequence to a file (called by subclasses in apply method)"""
        if self.processSequence is not None:
            processFilePath = os.path.join(self.runDir, f"{self.name}-processSequence.py")
            source = inspect.getsource(self.processSequence)
            with open(processFilePath, "w") as f:
                f.write("import viennaps2d as vps\n")
                f.write("import viennals2d as vls\n\n")
                f.write(source)
            print(f"Process sequence saved to: {processFilePath}")

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

    def setDistanceMetric(self, metric: str, criticalDimensionRanges: list[dict] = None):
        """
        Set the distance metric for comparing level sets.

        Args:
            metric: Distance metric to be used.
                    Options include:
                        - 'CA' (Compare area mismatch),
                        - 'CSF' (Compare sparse field),
                        - 'CNB' (Compare narrow band),
                        - 'CA+CSF' (Sum of CA and CSF),
                        - 'CA+CNB' (Sum of CA and CNB),
                        - 'CCD' (Compare critical dimensions).
            criticalDimensionRanges: Required only for 'CCD' metric. List of range configurations.
                    Each dict should have:
                        - 'axis': 'x' or 'y' (the axis to scan along)
                        - 'min': minimum value of the range
                        - 'max': maximum value of the range
                        - 'findMaximum': True to find maximum, False to find minimum
                    Example: [{'axis': 'x', 'min': -5, 'max': 5, 'findMaximum': True}]
        """
        if metric not in ["CA", "CSF", "CNB", "CA+CSF", "CA+CNB", "CCD"]:
            raise ValueError(
                f"Invalid distance metric: {metric}. "
                "Options are 'CA', 'CSF', 'CNB', 'CA+CSF', 'CA+CNB', 'CCD'."
            )

        if metric == "CCD" and criticalDimensionRanges is None:
            raise ValueError(
                "criticalDimensionRanges must be provided when using 'CCD' metric"
            )

        self.distanceMetric = metric
        self.criticalDimensionRanges = criticalDimensionRanges
        return self

    def apply(self, *args, **kwargs):
        """Apply the study - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement apply()")
