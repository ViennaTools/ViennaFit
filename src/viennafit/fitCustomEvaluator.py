from .fitProject import Project
from .fitUtilities import loadOptimumFromResultsFile
from .fitDistanceMetrics import DistanceMetric
import viennaps as vps
import viennals as vls
import importlib.util
import sys
import os
import json
import inspect
import time
import itertools
import shutil
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
        self.pairedVariableValues = None  # List[Dict[str, float]] - specific parameter combinations (not grid)
        self.distanceMetric = None
        self.distanceMetricFunction = None
        self.additionalDistanceMetrics = []  # List of additional metric names
        self.additionalDistanceMetricFunctions = {}  # Dict of metric name -> callable
        self.isMultiDomainProcess = False
        self.gridResults = []  # List of evaluation results
        self.evaluationName = None
        self.savedProcessSequencePath = None  # Path to saved process sequence in customEvaluations

        # Check project readiness
        if not project.isReady:
            raise ValueError(
                "Project is not ready. Please initialize the project first, "
                "set the initial and target domains before using the evaluator."
            )

    def getAvailableInitialDomains(self) -> List[str]:
        """Get list of available initial domain names."""
        domains = self.project.listInitialDomains()
        if self.project.initialDomain is not None and "default" not in domains:
            domains.append("single")  # Indicate single domain is available
        return domains

    def getAvailableTargetDomains(self) -> List[str]:
        """Get list of available target domain names."""
        return self.project.listTargetLevelSets()

    def validateMultiDomainSetup(self) -> List[str]:
        """Validate multi-domain setup and return list of issues."""
        issues = []

        if self.isMultiDomainProcess:
            # Check for multi-domain requirements
            if len(self.project.initialDomains) == 0:
                issues.append("Multi-domain process requires multiple initial domains")

            if len(self.project.targetLevelSets) == 0:
                issues.append("Multi-domain process requires multiple target domains")

            # Check domain name matching
            initialNames = set(self.project.initialDomains.keys())
            targetNames = set(self.project.targetLevelSets.keys())

            missingTargets = initialNames - targetNames
            if missingTargets:
                issues.append(
                    f"Missing target domains for initial domains: {', '.join(missingTargets)}"
                )
        else:
            # Check for single-domain requirements
            if (
                self.project.initialDomain is None
                and len(self.project.initialDomains) == 0
            ):
                issues.append("No initial domain available")

            if (
                self.project.targetLevelSet is None
                and len(self.project.targetLevelSets) == 0
            ):
                issues.append("No target domain available")

        return issues

    def isReadyForEvaluation(self) -> bool:
        """Check if evaluator is ready for grid evaluation."""
        if not self.processSequence:
            return False

        issues = self.validateMultiDomainSetup()
        return len(issues) == 0

    def loadOptimizationRun(
        self, runName: str, skipProcessSequence: bool = False
    ) -> "CustomEvaluator":
        """
        Load optimization results and process sequence from a completed run.

        Args:
            runName: Name of the optimization run to load
            skipProcessSequence: If True, skip loading the process sequence from the run
                               (useful when you want to use a custom sequence)

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

        # Load process sequence file unless skipped
        if not skipProcessSequence:
            processSequenceFile = os.path.join(runDir, f"{runName}-processSequence.py")
            if not os.path.exists(processSequenceFile):
                raise FileNotFoundError(
                    f"Process sequence file not found: {processSequenceFile}"
                )

            self.processSequencePath = processSequenceFile
            self._loadProcessSequence(processSequenceFile)

            # Detect if process sequence supports multi-domain processing
            self.isMultiDomainProcess = self._detectMultiDomainProcess()

        print(f"Loaded optimization run '{runName}':")
        print(f"  Best score: {results.get('bestScore', 'Unknown')}")
        print(
            f"  Parameters: {len(self.optimalParameters)} variable, {len(self.fixedParameters)} fixed"
        )
        if not skipProcessSequence:
            print(f"  Process sequence: {self.processSequence.__name__}")
            print(
                f"  Multi-domain support: {'Yes' if self.isMultiDomainProcess else 'No'}"
            )
        else:
            print(
                "  Process sequence: Skipped (use setProcessSequence() to set a custom one)"
            )

        return self

    def loadProcessSequenceFromFile(self, filePath: str) -> "CustomEvaluator":
        """
        Load a process sequence from a Python file.

        This allows you to override the process sequence loaded from an optimization run,
        or to set a process sequence without loading an optimization run.

        Args:
            filePath: Path to the Python file containing the process sequence function

        Returns:
            self: For method chaining
        """
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Process sequence file not found: {filePath}")

        self.processSequencePath = filePath
        self._loadProcessSequence(filePath)

        # Detect if process sequence supports multi-domain processing
        self.isMultiDomainProcess = self._detectMultiDomainProcess()

        print(f"Loaded process sequence from: {os.path.basename(filePath)}")
        print(f"  Function: {self.processSequence.__name__}")
        print(f"  Multi-domain support: {'Yes' if self.isMultiDomainProcess else 'No'}")

        return self

    def setProcessSequence(self, processSequence) -> "CustomEvaluator":
        """
        Set a custom process sequence function directly.

        This allows you to override the process sequence loaded from an optimization run,
        or to set a process sequence programmatically.

        Args:
            processSequence: Either a callable function with signature (Domain, dict[str, float])
                           or (dict[str, Domain], dict[str, float]) for multi-domain,
                           or a string file path to load the sequence from

        Returns:
            self: For method chaining

        Raises:
            ValueError: If the process sequence doesn't have the expected signature
        """
        # If it's a string, treat it as a file path
        if isinstance(processSequence, str):
            return self.loadProcessSequenceFromFile(processSequence)

        # Validate that it's callable
        if not callable(processSequence):
            raise ValueError(
                "Process sequence must be a callable function or a file path"
            )

        # Validate function signature
        try:
            sig = inspect.signature(processSequence)
            params = list(sig.parameters.values())

            if len(params) != 2:
                raise ValueError(
                    f"Process sequence must have exactly 2 parameters, got {len(params)}"
                )

            # Check parameter types (allow empty annotations for flexibility)
            firstParam = params[0]
            secondParam = params[1]

            # Valid first parameter types
            validFirstTypes = [
                vps.Domain,
                dict[str, vps.Domain],
                list[vps.Domain],
                inspect.Parameter.empty,
            ]

            # Valid second parameter types
            validSecondTypes = [dict[str, float], inspect.Parameter.empty]

            if firstParam.annotation not in validFirstTypes:
                print(
                    f"Warning: First parameter type annotation is {firstParam.annotation}, "
                    f"expected one of: Domain, dict[str, Domain], list[Domain]"
                )

            if secondParam.annotation not in validSecondTypes:
                print(
                    f"Warning: Second parameter type annotation is {secondParam.annotation}, "
                    f"expected: dict[str, float]"
                )

        except ValueError as e:
            raise ValueError(f"Invalid process sequence signature: {str(e)}")

        # Set the process sequence
        self.processSequence = processSequence
        self.processSequencePath = None  # Mark as not loaded from file

        # Detect if process sequence supports multi-domain processing
        self.isMultiDomainProcess = self._detectMultiDomainProcess()

        print(f"Set process sequence: {processSequence.__name__}")
        print(f"  Multi-domain support: {'Yes' if self.isMultiDomainProcess else 'No'}")

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

    def _detectMultiDomainProcess(self) -> bool:
        """Detect if the process sequence function supports multi-domain processing."""
        if self.processSequence is None:
            return False

        try:
            sig = inspect.signature(self.processSequence)
            params = list(sig.parameters.values())

            if len(params) >= 1:
                firstParam = params[0]
                # Check if first parameter is annotated as dict[str, Domain] or list[Domain]
                if firstParam.annotation in [dict[str, vps.Domain], list[vps.Domain]]:
                    return True
                # Also check if project has multiple domains/targets
                elif (
                    len(getattr(self.project, "initialDomains", {})) > 0
                    and len(getattr(self.project, "targetLevelSets", {})) > 0
                ):
                    return True

            return False
        except Exception:
            return False

    def setDistanceMetric(
        self, metric: str, criticalDimensionRanges: list[dict] = None
    ) -> "CustomEvaluator":
        """
        Set the distance metric for comparing level sets.

        Args:
            metric: Distance metric to be used.
                   Options: 'CA', 'CSF', 'CNB', 'CA+CSF', 'CA+CNB', 'CCD'
            criticalDimensionRanges: Required only for 'CCD' metric. List of range configurations.
                    Each dict should have:
                        - 'axis': 'x' or 'y' (the axis to scan along)
                        - 'min': minimum value of the range
                        - 'max': maximum value of the range
                        - 'findMaximum': True to find maximum, False to find minimum
                    Example: [{'axis': 'x', 'min': -5, 'max': 5, 'findMaximum': True}]

        Returns:
            self: For method chaining
        """
        if metric not in DistanceMetric.AVAILABLE_METRICS:
            raise ValueError(
                f"Invalid distance metric: {metric}. "
                f"Options are {', '.join(DistanceMetric.AVAILABLE_METRICS)}."
            )

        if metric == "CCD" and criticalDimensionRanges is None:
            raise ValueError(
                "criticalDimensionRanges must be provided when using 'CCD' metric"
            )

        self.distanceMetric = metric
        self.criticalDimensionRanges = criticalDimensionRanges
        # Create the distance metric function with appropriate multi-domain support
        self.distanceMetricFunction = DistanceMetric.create(
            metric,
            multiDomain=self.isMultiDomainProcess,
            criticalDimensionRanges=criticalDimensionRanges,
        )
        return self

    def setAdditionalMetrics(
        self,
        metrics: List[str],
        criticalDimensionRanges: list[dict] = None,
        sparseFieldExpansionWidth: int = 200,
    ) -> "CustomEvaluator":
        """
        Set additional distance metrics to evaluate alongside the primary metric.

        Args:
            metrics: List of distance metric names to evaluate
                    Options: 'CA', 'CSF', 'CSF-IS', 'CNB', 'CA+CSF', 'CA+CNB', 'CCD', 'CCH'
            criticalDimensionRanges: Required only for 'CCD' metric. List of range configurations.
            sparseFieldExpansionWidth: Expansion width for CSF and CSF-IS metrics (default: 200)

        Returns:
            self: For method chaining
        """
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list of metric names")

        self.additionalDistanceMetrics = []
        self.additionalDistanceMetricFunctions = {}

        for metric in metrics:
            if metric not in DistanceMetric.AVAILABLE_METRICS:
                raise ValueError(
                    f"Invalid distance metric: {metric}. "
                    f"Options are {', '.join(DistanceMetric.AVAILABLE_METRICS)}."
                )

            if metric == "CCD" and criticalDimensionRanges is None:
                raise ValueError(
                    "criticalDimensionRanges must be provided when using 'CCD' metric"
                )

            # Store metric name
            self.additionalDistanceMetrics.append(metric)

            # Create metric function
            self.additionalDistanceMetricFunctions[metric] = DistanceMetric.create(
                metric,
                multiDomain=self.isMultiDomainProcess,
                criticalDimensionRanges=criticalDimensionRanges,
                sparseFieldExpansionWidth=sparseFieldExpansionWidth,
            )

        print(f"Set {len(self.additionalDistanceMetrics)} additional metrics: {', '.join(self.additionalDistanceMetrics)}")
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

    def setVariableValuesPaired(
        self, pairedValues: List[Dict[str, float]]
    ) -> "CustomEvaluator":
        """
        Set specific parameter combinations to evaluate (not a full grid).

        This method allows you to specify exact parameter combinations to evaluate,
        rather than generating all combinations from lists (Cartesian product).

        Args:
            pairedValues: List of parameter dictionaries, each representing one evaluation.
                         All dictionaries must have the same keys.
                         Example: [
                             {"etchantFlux": 182.4, "ionFlux": 25.7, "ptDepth": 1.02},
                             {"etchantFlux": 364.8, "ionFlux": 51.5, "ptDepth": 2.04},
                             ...
                         ]

        Returns:
            self: For method chaining

        Raises:
            ValueError: If pairedValues is empty or dictionaries have inconsistent keys
        """
        if not pairedValues or len(pairedValues) == 0:
            raise ValueError("pairedValues must be a non-empty list")

        # Validate that all dictionaries have the same keys
        firstKeys = set(pairedValues[0].keys())
        for i, combo in enumerate(pairedValues[1:], start=1):
            if set(combo.keys()) != firstKeys:
                raise ValueError(
                    f"Inconsistent parameter keys in pairedValues[{i}]. "
                    f"Expected {firstKeys}, got {set(combo.keys())}"
                )

        # Validate that all parameters exist in optimal or fixed parameters
        allParams = {**self.fixedParameters, **self.optimalParameters}
        for paramName in firstKeys:
            if paramName not in allParams:
                print(
                    f"Warning: Parameter '{paramName}' not found in optimization results"
                )

        self.pairedVariableValues = pairedValues
        # Clear regular variable values to avoid confusion
        self.variableValues = {}

        print(f"Set {len(pairedValues)} specific parameter combinations (not a grid)")
        print(f"Parameters: {', '.join(sorted(firstKeys))}")
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
        self,
        evaluationName: str,
        saveVisualization: bool = True,
        initialDomainName: str = None,
        saveAllMetricVisualizations: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all combinations of variable values in a grid.

        Args:
            evaluationName: Name for this custom evaluation run
            saveVisualization: Whether to save visualization files for primary metric
            initialDomainName: Name of the initial domain to use (default uses single domain)
            saveAllMetricVisualizations: Whether to save visualization files for all additional metrics

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

        # Ensure distance metric function is created if not already done
        if not self.distanceMetricFunction:
            criticalDimensionRanges = getattr(self, "criticalDimensionRanges", None)
            self.distanceMetricFunction = DistanceMetric.create(
                self.distanceMetric,
                multiDomain=self.isMultiDomainProcess,
                criticalDimensionRanges=criticalDimensionRanges,
            )

        if not self.variableValues and not self.pairedVariableValues:
            raise ValueError("No variable values set. Call setVariableValues() or setVariableValuesPaired() first.")

        # Validate multi-domain setup
        validationIssues = self.validateMultiDomainSetup()
        if validationIssues:
            raise ValueError(
                f"Multi-domain setup validation failed: {'; '.join(validationIssues)}"
            )

        self.evaluationName = evaluationName
        self.gridResults = []

        # Create output directory
        outputDir = os.path.join(
            self.project.projectPath, "customEvaluations", evaluationName
        )
        os.makedirs(outputDir, exist_ok=True)

        # Save the process sequence to the output directory
        self.savedProcessSequencePath = None
        processSequenceDestPath = os.path.join(
            outputDir, f"{evaluationName}-processSequence.py"
        )

        try:
            if self.processSequencePath and os.path.exists(self.processSequencePath):
                # Copy process sequence file from original location
                shutil.copy2(self.processSequencePath, processSequenceDestPath)
                self.savedProcessSequencePath = processSequenceDestPath
                print(
                    f"Saved process sequence to: {os.path.basename(processSequenceDestPath)}"
                )
            elif self.processSequence:
                # Process sequence was set directly as a callable, extract source
                try:
                    processSequenceSource = inspect.getsource(self.processSequence)
                    with open(processSequenceDestPath, "w") as f:
                        f.write(processSequenceSource)
                    self.savedProcessSequencePath = processSequenceDestPath
                    print(
                        f"Saved process sequence source to: {os.path.basename(processSequenceDestPath)}"
                    )
                except (OSError, TypeError) as e:
                    # Could not extract source (e.g., built-in function, lambda in REPL)
                    warningPath = os.path.join(
                        outputDir, f"{evaluationName}-processSequence-WARNING.txt"
                    )
                    with open(warningPath, "w") as f:
                        f.write(
                            f"Warning: Could not extract process sequence source code.\n"
                        )
                        f.write(f"Function name: {self.processSequence.__name__}\n")
                        f.write(f"Error: {str(e)}\n\n")
                        f.write(
                            "The process sequence was set programmatically and its source "
                            "code could not be retrieved.\n"
                        )
                    print(
                        f"Warning: Could not extract process sequence source. See {os.path.basename(warningPath)}"
                    )
        except Exception as e:
            print(
                f"Warning: Failed to save process sequence: {str(e)}. Continuing with evaluation..."
            )

        # Generate parameter combinations
        if self.pairedVariableValues is not None:
            # Use paired values (specific combinations, not grid)
            paramNames = list(self.pairedVariableValues[0].keys())
            combinations = []
            for combo in self.pairedVariableValues:
                # Convert dict to tuple in same order as paramNames
                combinations.append(tuple(combo[name] for name in paramNames))
            print(
                f"Starting paired evaluation '{evaluationName}' with {len(combinations)} specific combinations..."
            )
        else:
            # Use grid (Cartesian product)
            paramNames = list(self.variableValues.keys())
            paramValueLists = [self.variableValues[name] for name in paramNames]
            combinations = list(itertools.product(*paramValueLists))
            print(
                f"Starting grid evaluation '{evaluationName}' with {len(combinations)} combinations..."
            )

        print(
            f"Multi-domain mode: {'Enabled' if self.isMultiDomainProcess else 'Disabled'}"
        )
        if self.isMultiDomainProcess:
            print(
                f"Available initial domains: {list(self.project.initialDomains.keys())}"
            )
            print(
                f"Available target domains: {list(self.project.targetLevelSets.keys())}"
            )

        distanceFunction = self.distanceMetricFunction

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
                # Generate output name for this evaluation
                outputName = f"eval_{i:04d}"
                writePath = None
                if saveVisualization:
                    writePath = os.path.join(outputDir, outputName)

                if self.isMultiDomainProcess:
                    # Multi-domain processing
                    initialDomains = {}

                    if initialDomainName is not None:
                        # Use specific named domain only
                        if initialDomainName not in self.project.initialDomains:
                            raise ValueError(
                                f"Initial domain '{initialDomainName}' not found in project"
                            )
                        domainCopy = vps.Domain(
                            self.project.initialDomains[initialDomainName]
                        )
                        initialDomains[initialDomainName] = domainCopy
                    else:
                        # Use all available initial domains
                        for name, domain in self.project.initialDomains.items():
                            domainCopy = vps.Domain(domain)
                            initialDomains[name] = domainCopy

                    # Run process sequence with all domains
                    resultDomains = self.processSequence(initialDomains, evalParams)

                    if not isinstance(resultDomains, dict):
                        raise ValueError(
                            "Multi-domain process sequence must return dict[str, Domain]"
                        )

                    # Convert result domains to level sets for comparison
                    resultLevelSets = {}
                    for name, domain in resultDomains.items():
                        if domain.getLevelSets():
                            resultLevelSets[name] = domain.getLevelSets()[
                                -1
                            ]  # Use last level set
                        else:
                            raise ValueError(
                                f"Result domain '{name}' has no level sets"
                            )

                    # Save visualization if requested
                    resultPaths = {}
                    resultPath = None
                    if saveVisualization:
                        for name, levelSet in resultLevelSets.items():
                            resultPath = f"{writePath}-result-{name}.vtp"
                            resultMesh = vls.Mesh()
                            vls.ToSurfaceMesh(levelSet, resultMesh).apply()
                            vls.VTKWriter(resultMesh, resultPath).apply()
                            resultPaths[name] = resultPath

                    # Calculate objective value using multi-domain distance metric
                    primaryMetricStartTime = time.time()
                    objectiveValue = distanceFunction(
                        resultLevelSets,
                        self.project.targetLevelSets,
                        saveVisualization,
                        writePath,
                    )
                    primaryMetricTime = time.time() - primaryMetricStartTime

                    # Calculate additional metrics
                    additionalMetricValues = {}
                    additionalMetricTimes = {}
                    for metricName, metricFunc in self.additionalDistanceMetricFunctions.items():
                        additionalMetricStartTime = time.time()
                        additionalMetricValues[metricName] = metricFunc(
                            resultLevelSets,
                            self.project.targetLevelSets,
                            saveAllMetricVisualizations,  # Save visualization if requested
                            writePath,
                        )
                        additionalMetricTimes[metricName] = time.time() - additionalMetricStartTime

                else:
                    # Single-domain processing (backward compatibility)
                    if initialDomainName is not None:
                        # Use named initial domain
                        if initialDomainName not in self.project.initialDomains:
                            raise ValueError(
                                f"Initial domain '{initialDomainName}' not found in project"
                            )
                        domainCopy = vps.Domain(
                            self.project.initialDomains[initialDomainName]
                        )
                    else:
                        # Use default single initial domain for backward compatibility
                        domainCopy = vps.Domain(self.project.initialDomain)

                    # Run process sequence with current parameters on the copy
                    resultDomain = self.processSequence(domainCopy, evalParams)

                    # Save visualization if requested
                    resultPath = None
                    resultPaths = None
                    if saveVisualization:
                        # Save result domain
                        resultPath = f"{writePath}-result.vtp"
                        resultMesh = vls.Mesh()
                        vls.ToSurfaceMesh(resultDomain, resultMesh).apply()
                        vls.VTKWriter(resultMesh, resultPath).apply()

                    # Calculate objective value using single-domain distance metric
                    primaryMetricStartTime = time.time()
                    objectiveValue = distanceFunction(
                        resultDomain,
                        self.project.targetLevelSet,
                        saveVisualization,
                        writePath,
                    )
                    primaryMetricTime = time.time() - primaryMetricStartTime

                    # Calculate additional metrics
                    additionalMetricValues = {}
                    additionalMetricTimes = {}
                    for metricName, metricFunc in self.additionalDistanceMetricFunctions.items():
                        additionalMetricStartTime = time.time()
                        additionalMetricValues[metricName] = metricFunc(
                            resultDomain,
                            self.project.targetLevelSet,
                            saveAllMetricVisualizations,  # Save visualization if requested
                            writePath,
                        )
                        additionalMetricTimes[metricName] = time.time() - additionalMetricStartTime

                executionTime = time.time() - startTime

                # Store result
                result = {
                    "evaluationNumber": i,
                    "parameters": dict(zip(paramNames, combination)),
                    "allParameters": evalParams,
                    "objectiveValue": objectiveValue,
                    "executionTime": executionTime,
                    "primaryMetricTime": primaryMetricTime,
                    "additionalMetricValues": additionalMetricValues,
                    "additionalMetricTimes": additionalMetricTimes,
                    "resultPath": (
                        resultPaths if self.isMultiDomainProcess else resultPath
                    ),
                    "multiDomain": self.isMultiDomainProcess,
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
                    "primaryMetricTime": 0.0,
                    "additionalMetricValues": {},
                    "additionalMetricTimes": {},
                    "error": str(e),
                    "resultPath": None,
                    "multiDomain": self.isMultiDomainProcess,
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
                "processSequenceSaved": (
                    os.path.basename(self.savedProcessSequencePath)
                    if self.savedProcessSequencePath
                    else None
                ),
                "processSequenceSavedSuccessfully": self.savedProcessSequencePath is not None,
                "distanceMetric": self.distanceMetric,
                "additionalDistanceMetrics": self.additionalDistanceMetrics,
                "totalEvaluations": len(self.gridResults),
                "successfulEvaluations": len(validResults),
                "failedEvaluations": len(self.gridResults) - len(validResults),
            },
            "configuration": {
                "optimalParameters": self.optimalParameters,
                "fixedParameters": self.fixedParameters,
                "variableValues": self.variableValues,
                "isMultiDomainProcess": self.isMultiDomainProcess,
                "availableInitialDomains": self.getAvailableInitialDomains(),
                "availableTargetDomains": self.getAvailableTargetDomains(),
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
                headers = ["evaluationNumber"] + paramNames

                # Add primary metric columns
                if self.distanceMetric:
                    headers.extend([f"{self.distanceMetric}_value", f"{self.distanceMetric}_time"])
                else:
                    headers.extend(["objectiveValue", "primaryMetricTime"])

                # Add additional metric columns
                for metricName in self.additionalDistanceMetrics:
                    headers.extend([f"{metricName}_value", f"{metricName}_time"])

                # Add execution time at the end
                headers.append("executionTime")

                f.write(",".join(headers) + "\n")

                # Write data rows
                for result in self.gridResults:
                    row = [str(result["evaluationNumber"])]
                    for paramName in paramNames:
                        row.append(str(result["parameters"].get(paramName, "")))

                    # Add primary metric value and time
                    row.append(str(result.get("objectiveValue", "")))
                    row.append(str(result.get("primaryMetricTime", "")))

                    # Add additional metric values and times
                    additionalMetricValues = result.get("additionalMetricValues", {})
                    additionalMetricTimes = result.get("additionalMetricTimes", {})
                    for metricName in self.additionalDistanceMetrics:
                        row.append(str(additionalMetricValues.get(metricName, "")))
                        row.append(str(additionalMetricTimes.get(metricName, "")))

                    # Add execution time
                    row.append(str(result["executionTime"]))
                    f.write(",".join(row) + "\n")

        print(f"Grid evaluation report saved to: {reportPath}")
        print(f"CSV summary saved to: {csvPath}")

        return reportPath
