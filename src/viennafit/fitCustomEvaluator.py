from .fitProject import Project
from .fitUtilities import loadOptimumFromResultsFile
from .fitDistanceMetrics import DistanceMetric
import viennaps2d as vps
import viennals2d as vls
import importlib.util
import sys
import os
import json
import inspect
import time
import itertools
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy


class CustomEvaluator:
    """
    Evaluates process sequences with grid-based parameter variations from optimization results.

    This class loads optimal parameters from completed optimization runs and allows
    for systematic evaluation across parameter grids, keeping optimal values for
    non-specified parameters.
    """

    def __init__(self, project: Project):
        """
        Initialize the CustomEvaluator.

        Args:
            project: ViennaFit project instance
        """
        self.project = project
        self.optimizationResultsPath = None
        self.processSequencePath = None
        self.processSequence = None
        self.optimalParameters = {}
        self.fixedParameters = {}
        self.variableValues = (
            {}
        )  # Dict[str, List[float]] - parameter name to list of values
        self.distanceMetric = None
        self.gridResults = []  # List of evaluation results
        self.evaluationName = None

        # Check project readiness
        if not project.isReady:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains before using the evaluator."
            )

    def loadOptimizationRun(self, runName: str) -> "CustomEvaluator":
        """
        Load optimization results and process sequence from a completed run.

        Args:
            runName: Name of the optimization run to load

        Returns:
            self: For method chaining
        """
        # Find the optimization run directory
        runDir = os.path.join(self.project.projectPath, "optimizationRuns", runName)
        if not os.path.exists(runDir):
            raise FileNotFoundError(
                f"Optimization run '{runName}' not found in project"
            )

        # Load results file
        resultsFile = os.path.join(runDir, f"{runName}-final-results.json")
        if not os.path.exists(resultsFile):
            raise FileNotFoundError(f"Results file not found: {resultsFile}")

        # Load optimization results
        results = loadOptimumFromResultsFile(resultsFile)
        self.optimizationResultsPath = resultsFile
        self.optimalParameters = deepcopy(results["bestParameters"])

        # Store fixed parameters if available
        if "fixedParameters" in results:
            self.fixedParameters = deepcopy(results["fixedParameters"])

        # Load process sequence file
        processSequenceFile = os.path.join(runDir, f"{runName}-processSequence.py")
        if not os.path.exists(processSequenceFile):
            raise FileNotFoundError(
                f"Process sequence file not found: {processSequenceFile}"
            )

        self.processSequencePath = processSequenceFile
        self._loadProcessSequence(processSequenceFile)

        print(f"Loaded optimization run '{runName}':")
        print(f"  Best score: {results.get('bestScore', 'Unknown')}")
        print(
            f"  Parameters: {len(self.optimalParameters)} variable, {len(self.fixedParameters)} fixed"
        )
        print(f"  Process sequence: {self.processSequence.__name__}")

        return self

    def _loadProcessSequence(self, filePath: str):
        """Load a process sequence from a Python file"""
        absPath = os.path.abspath(filePath)
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
                        # Check parameter types (allow empty annotations)
                        if (
                            params[0].annotation == vps.Domain
                            or params[0].annotation == inspect.Parameter.empty
                        ):
                            if (
                                params[1].annotation == dict[str, float]
                                or params[1].annotation == inspect.Parameter.empty
                            ):
                                self.processSequence = item
                                return
                except ValueError:
                    continue

        raise ValueError(
            f"No suitable process sequence function found in file: {absPath}"
        )

    def setDistanceMetric(self, metric: str) -> "CustomEvaluator":
        """
        Set the distance metric for comparing level sets.

        Args:
            metric: Distance metric to be used.
                   Options: 'CA', 'CSF', 'CNB', 'CA+CSF', 'CA+CNB'

        Returns:
            self: For method chaining
        """
        if metric not in DistanceMetric.AVAILABLE_METRICS:
            raise ValueError(
                f"Invalid distance metric: {metric}. "
                f"Options are {', '.join(DistanceMetric.AVAILABLE_METRICS)}."
            )

        self.distanceMetric = metric
        return self

    def setVariableValues(
        self, variableValues: Dict[str, List[float]]
    ) -> "CustomEvaluator":
        """
        Set lists of values for parameters to evaluate in grid.

        Args:
            variableValues: Dictionary mapping parameter names to lists of values to evaluate

        Returns:
            self: For method chaining
        """
        for paramName, valueList in variableValues.items():
            if paramName not in self.optimalParameters:
                if paramName in self.fixedParameters:
                    print(f"Warning: '{paramName}' is a fixed parameter, cannot vary")
                    continue
                else:
                    print(
                        f"Warning: Parameter '{paramName}' not found in optimization results"
                    )
                    continue

            if not isinstance(valueList, list) or len(valueList) == 0:
                raise ValueError(
                    f"Values for parameter '{paramName}' must be a non-empty list"
                )

            self.variableValues[paramName] = valueList
            print(f"Set variable values for '{paramName}': {len(valueList)} values")

        totalCombinations = 1
        for values in self.variableValues.values():
            totalCombinations *= len(values)

        print(f"Grid will evaluate {totalCombinations} parameter combinations")
        return self

    def getOptimalParameters(self) -> Dict[str, float]:
        """
        Get optimal parameter values from the optimization run.

        Returns:
            Dictionary of optimal parameter values (variable + fixed)
        """
        allParams = deepcopy(self.fixedParameters)
        allParams.update(self.optimalParameters)
        return allParams

    def evaluateGrid(
        self, evaluationName: str, saveVisualization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all combinations of variable values in a grid.

        Args:
            evaluationName: Name for this custom evaluation run
            saveVisualization: Whether to save visualization files

        Returns:
            List of evaluation results for each parameter combination
        """
        if not self.processSequence:
            raise ValueError(
                "No process sequence loaded. Call loadOptimizationRun() first."
            )

        if not self.distanceMetric:
            print(
                "Warning: No distance metric set. Using 'CA' (Compare Area) by default."
            )
            self.setDistanceMetric("CA")

        if not self.variableValues:
            raise ValueError("No variable values set. Call setVariableValues() first.")

        self.evaluationName = evaluationName
        self.gridResults = []

        # Create output directory
        outputDir = os.path.join(
            self.project.projectPath, "customEvaluations", evaluationName
        )
        os.makedirs(outputDir, exist_ok=True)

        # Generate all parameter combinations
        paramNames = list(self.variableValues.keys())
        paramValueLists = [self.variableValues[name] for name in paramNames]
        combinations = list(itertools.product(*paramValueLists))

        print(
            f"Starting grid evaluation '{evaluationName}' with {len(combinations)} combinations..."
        )

        distanceFunction = DistanceMetric.create(self.distanceMetric)

        for i, combination in enumerate(combinations, 1):
            # Create parameter set for this combination
            evalParams = deepcopy(self.fixedParameters)
            evalParams.update(self.optimalParameters)  # Start with optimal values

            # Override with current combination values
            for paramName, value in zip(paramNames, combination):
                evalParams[paramName] = value

            print(f"\nEvaluation {i}/{len(combinations)}:")
            print("  Variable parameters:")
            for paramName, value in zip(paramNames, combination):
                optimal = self.optimalParameters[paramName]
                print(f"    {paramName}: {value:.6f} (optimal: {optimal:.6f})")

            # Execute process sequence
            startTime = time.time()

            try:
                # Create a fresh copy of the initial domain for each evaluation
                domainCopy = vps.Domain(self.project.initialDomain)
                
                # Run process sequence with current parameters on the copy
                resultDomain = self.processSequence(domainCopy, evalParams)

                # Generate output name for this evaluation
                outputName = f"eval_{i:04d}"

                # Save visualization if requested
                writePath = None
                resultPath = None
                if saveVisualization:
                    writePath = os.path.join(outputDir, outputName)

                    # Save result domain
                    resultPath = f"{writePath}-result.vtp"
                    resultMesh = vls.Mesh()
                    vls.ToSurfaceMesh(resultDomain, resultMesh).apply()
                    vls.VTKWriter(resultMesh, resultPath).apply()

                # Calculate objective value
                objectiveValue = distanceFunction(
                    resultDomain,
                    self.project.targetLevelSet,
                    saveVisualization,
                    writePath,
                )

                executionTime = time.time() - startTime

                # Store result
                result = {
                    "evaluationNumber": i,
                    "parameters": dict(zip(paramNames, combination)),
                    "allParameters": evalParams,
                    "objectiveValue": objectiveValue,
                    "executionTime": executionTime,
                    "resultPath": resultPath,
                }

                self.gridResults.append(result)

                print(f"  Objective value: {objectiveValue:.6f}")
                print(f"  Execution time: {executionTime:.2f} seconds")

            except Exception as e:
                print(f"  Error in evaluation {i}: {str(e)}")
                # Store error result
                result = {
                    "evaluationNumber": i,
                    "parameters": dict(zip(paramNames, combination)),
                    "allParameters": evalParams,
                    "objectiveValue": float("inf"),
                    "executionTime": time.time() - startTime,
                    "error": str(e),
                    "resultPath": None,
                }
                self.gridResults.append(result)

        # Save grid results
        self.saveGridReport(outputDir)

        print(f"\nGrid evaluation completed: {len(self.gridResults)} evaluations")
        print(f"Results saved to: {outputDir}")

        return self.gridResults

    def getGridResults(self) -> List[Dict[str, Any]]:
        """
        Get the results from the last grid evaluation.

        Returns:
            List of evaluation results
        """
        return deepcopy(self.gridResults)

    def getBestResult(self) -> Optional[Dict[str, Any]]:
        """
        Get the best result from the grid evaluation.

        Returns:
            Dictionary with the best evaluation result, or None if no results
        """
        if not self.gridResults:
            return None

        validResults = [
            r for r in self.gridResults if r["objectiveValue"] != float("inf")
        ]
        if not validResults:
            return None

        return min(validResults, key=lambda x: x["objectiveValue"])

    def saveGridReport(self, outputDir: str = None) -> str:
        """
        Save a comprehensive report of grid evaluation results.

        Args:
            outputDir: Directory to save the report (default: last evaluation directory)

        Returns:
            Path to the saved report file
        """
        if outputDir is None and self.evaluationName:
            outputDir = os.path.join(
                self.project.projectPath, "customEvaluations", self.evaluationName
            )

        if outputDir is None:
            raise ValueError("No output directory specified and no evaluation name set")

        os.makedirs(outputDir, exist_ok=True)

        # Create comprehensive report
        bestResult = self.getBestResult()
        validResults = [
            r for r in self.gridResults if r["objectiveValue"] != float("inf")
        ]

        report = {
            "metadata": {
                "evaluationName": self.evaluationName,
                "projectName": self.project.projectName,
                "evaluationTime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimizationRunSource": (
                    os.path.basename(self.optimizationResultsPath)
                    if self.optimizationResultsPath
                    else None
                ),
                "processSequenceSource": (
                    os.path.basename(self.processSequencePath)
                    if self.processSequencePath
                    else None
                ),
                "distanceMetric": self.distanceMetric,
                "totalEvaluations": len(self.gridResults),
                "successfulEvaluations": len(validResults),
                "failedEvaluations": len(self.gridResults) - len(validResults),
            },
            "configuration": {
                "optimalParameters": self.optimalParameters,
                "fixedParameters": self.fixedParameters,
                "variableValues": self.variableValues,
            },
            "summary": {
                "bestObjectiveValue": (
                    bestResult["objectiveValue"] if bestResult else None
                ),
                "bestParameters": bestResult["parameters"] if bestResult else None,
                "bestEvaluationNumber": (
                    bestResult["evaluationNumber"] if bestResult else None
                ),
                "objectiveValueRange": {
                    "min": (
                        min(r["objectiveValue"] for r in validResults)
                        if validResults
                        else None
                    ),
                    "max": (
                        max(r["objectiveValue"] for r in validResults)
                        if validResults
                        else None
                    ),
                    "mean": (
                        sum(r["objectiveValue"] for r in validResults)
                        / len(validResults)
                        if validResults
                        else None
                    ),
                },
            },
            "results": self.gridResults,
        }

        # Save main report
        reportPath = os.path.join(outputDir, "grid_evaluation_report.json")
        with open(reportPath, "w") as f:
            json.dump(report, f, indent=2)

        # Save CSV summary for easy analysis
        csvPath = os.path.join(outputDir, "grid_results_summary.csv")
        with open(csvPath, "w") as f:
            if self.gridResults:
                # Write header
                paramNames = list(self.variableValues.keys())
                headers = (
                    ["evaluationNumber"]
                    + paramNames
                    + ["objectiveValue", "executionTime"]
                )
                f.write(",".join(headers) + "\n")

                # Write data rows
                for result in self.gridResults:
                    row = [str(result["evaluationNumber"])]
                    for paramName in paramNames:
                        row.append(str(result["parameters"].get(paramName, "")))
                    row.append(str(result["objectiveValue"]))
                    row.append(str(result["executionTime"]))
                    f.write(",".join(row) + "\n")

        print(f"Grid evaluation report saved to: {reportPath}")
        print(f"CSV summary saved to: {csvPath}")

        return reportPath
