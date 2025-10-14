from typing import Tuple, Callable
from viennaps2d import Domain
from .fitUtilities import (
    saveEvalToProgressFile,
    saveEvalToProgressManager,
)
from .fitDistanceMetrics import DistanceMetric
import time
import os
import inspect


class ObjectiveWrapper:
    """Factory class for creating objective function wrappers."""

    @staticmethod
    def create(optimizer: str, optimization, initialDomainName: str = None) -> Callable:
        """
        Create an objective function wrapper based on optimizer type.

        Args:
            optimizer: String identifying the optimizer ("dlib", etc)
            optimization: Reference to the Optimization instance
            initialDomainName: Name of the initial domain to use (optional)

        Returns:
            Callable: Wrapped objective function compatible with chosen optimizer
        """
        if optimizer == "dlib":
            wrapper = DlibObjectiveWrapper(optimization, initialDomainName)
        elif optimizer == "nevergrad":
            wrapper = NevergradObjectiveWrapper(optimization, initialDomainName)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Create a regular function that calls the wrapper
        def wrappedObjectiveFunction(*args):
            return wrapper(*args)

        return wrappedObjectiveFunction


class BaseObjectiveWrapper:
    """Base class for objective function wrappers."""

    def __init__(self, study, initialDomainName: str = None):
        self.study = study
        self.initialDomainName = initialDomainName

        # Detect if process sequence supports multi-domain processing
        self.isMultiDomainProcess = self._detectMultiDomainProcess()

        # Create appropriate distance metric(s)
        criticalDimensionRanges = getattr(study, "criticalDimensionRanges", None)
        sparseFieldExpansionWidth = getattr(study, "sparseFieldExpansionWidth", 200)

        # Primary metric (for optimization)
        primaryMetric = getattr(study, "primaryDistanceMetric", None) or study.distanceMetric
        self.distanceMetric = DistanceMetric.create(
            primaryMetric,
            multiDomain=self.isMultiDomainProcess,
            criticalDimensionRanges=criticalDimensionRanges,
            sparseFieldExpansionWidth=sparseFieldExpansionWidth,
        )

        # Additional metrics (for tracking)
        self.additionalDistanceMetrics = {}
        additionalMetrics = getattr(study, "additionalDistanceMetrics", [])
        for metricName in additionalMetrics:
            self.additionalDistanceMetrics[metricName] = DistanceMetric.create(
                metricName,
                multiDomain=self.isMultiDomainProcess,
                criticalDimensionRanges=criticalDimensionRanges,
                sparseFieldExpansionWidth=sparseFieldExpansionWidth,
            )

        self.progressManager = getattr(study, "progressManager", None)

    def _saveEvaluationData(
        self,
        paramValues: list[float],
        elapsedTime: float,
        objectiveValue: float,
        filename: str,
        isBest: bool = False,
        simulationTime: float = 0.0,
        distanceMetricTime: float = 0.0,
        additionalMetricValues: dict[str, float] = None,
        additionalMetricTimes: dict[str, float] = None,
    ):
        """Save evaluation data to progress file."""
        # Use new progress manager if available
        if self.progressManager:
            saveEvalToProgressManager(
                self.progressManager,
                evaluationNumber=self.study.evalCounter,
                parameterValues=paramValues,
                elapsedTime=elapsedTime,
                objectiveValue=objectiveValue,
                isBest=isBest,
                saveAll=True,
                simulationTime=simulationTime,
                distanceMetricTime=distanceMetricTime,
                additionalMetricValues=additionalMetricValues,
                additionalMetricTimes=additionalMetricTimes,
            )
        else:
            # Fallback to legacy system
            saveEvalToProgressFile(
                [*paramValues, elapsedTime, objectiveValue],
                os.path.join(self.study.runDir, filename + ".txt"),
            )

    def _evaluateObjective(
        self,
        paramDict: dict[str, float],
        saveVisualization: bool = False,
        saveAll: bool = False,
        variableParamNames: set[str] = None,
    ) -> Tuple[float, float, float, float]:
        """Run process sequence and evaluate result.

        Returns:
            Tuple[float, float, float, float]: (objectiveValue, elapsedTime, simulationTime, distanceMetricTime)
        """
        startTime = time.time()
        simulationStartTime = startTime

        if self.isMultiDomainProcess:
            # Multi-domain processing
            initialDomains = {}
            processedDomains = {}  # Keep references to vps.Domains that get processed

            if len(self.study.project.initialDomains) > 0:
                # Use multiple named initial domains
                for name, domain in self.study.project.initialDomains.items():
                    domainCopy = Domain()
                    domainCopy.deepCopy(domain)
                    initialDomains[name] = domainCopy
                    processedDomains[name] = domainCopy  # Same object, will be modified by process
            elif self.study.project.initialDomain is not None:
                # Fall back to single domain as "default"
                domainCopy = Domain()
                domainCopy.deepCopy(self.study.project.initialDomain)
                initialDomains["default"] = domainCopy
                processedDomains["default"] = domainCopy  # Same object, will be modified by process
            else:
                raise ValueError(
                    "No initial domains available for multi-domain processing"
                )

            # Apply process sequence to all domains (modifies vps.Domains in-place)
            processResult = self.study.processSequence(initialDomains, paramDict)

            # Handle optional timing dict return
            postProcessingTime = 0.0
            if isinstance(processResult, tuple) and len(processResult) == 2:
                resultDomains, timingDict = processResult
                postProcessingTime = timingDict.get("postProcessingTime", 0.0)
            else:
                resultDomains = processResult

            if not isinstance(resultDomains, dict):
                raise ValueError(
                    "Multi-domain process sequence must return dict[str, Domain] or (dict[str, Domain], dict)"
                )
        else:
            # Single-domain processing (backward compatibility)
            domainCopy = Domain()
            if self.initialDomainName is not None:
                # Use named initial domain
                if self.initialDomainName not in self.study.project.initialDomains:
                    raise ValueError(
                        f"Initial domain '{self.initialDomainName}' not found in project"
                    )
                domainCopy.deepCopy(
                    self.study.project.initialDomains[self.initialDomainName]
                )
            else:
                # Use default single initial domain for backward compatibility
                domainCopy.deepCopy(self.study.project.initialDomain)

            # Apply process sequence
            processResult = self.study.processSequence(domainCopy, paramDict)

            # Handle optional timing dict return
            postProcessingTime = 0.0
            if isinstance(processResult, tuple) and len(processResult) == 2:
                resultDomain, timingDict = processResult
                postProcessingTime = timingDict.get("postProcessingTime", 0.0)
            else:
                resultDomain = processResult

        # Track simulation time (subtract post-processing if provided)
        simulationTime = time.time() - simulationStartTime - postProcessingTime
        distanceMetricStartTime = time.time()

        self.study.evalCounter += 1

        # Evaluate primary metric and additional metrics with individual timing
        additionalMetricValues = {}
        additionalMetricTimes = {}

        if self.isMultiDomainProcess:
            # Multi-domain distance calculation
            if len(self.study.project.targetLevelSets) == 0:
                raise ValueError(
                    "No target level sets available for multi-domain comparison"
                )

            # Calculate primary objective value
            primaryMetricStartTime = time.time()
            objectiveValue = self.distanceMetric(
                resultDomains,
                self.study.project.targetLevelSets,
                False,  # Don't save visualization yet
                os.path.join(
                    self.study.progressDir,
                    f"{self.study.name}-{self.study.evalCounter}",
                ),
            )
            primaryMetricTime = time.time() - primaryMetricStartTime

            # Store primary metric value and time for tracking
            primaryMetricName = getattr(self.study, "primaryDistanceMetric", None) or self.study.distanceMetric
            additionalMetricValues[primaryMetricName] = objectiveValue
            additionalMetricTimes[primaryMetricName] = primaryMetricTime

            # Calculate additional metrics with individual timing
            for metricName, metricFunc in self.additionalDistanceMetrics.items():
                additionalMetricStartTime = time.time()
                additionalMetricValues[metricName] = metricFunc(
                    resultDomains,
                    self.study.project.targetLevelSets,
                    False,  # Don't save visualization
                    os.path.join(
                        self.study.progressDir,
                        f"{self.study.name}-{self.study.evalCounter}",
                    ),
                )
                additionalMetricTimes[metricName] = time.time() - additionalMetricStartTime

            # Only save visualization if saveAll is True or if current evaluation is better than current best
            if saveVisualization and (
                saveAll or objectiveValue <= self.study.bestScore
            ):
                # Save primary metric visualization
                self.distanceMetric(
                    resultDomains,
                    self.study.project.targetLevelSets,
                    True,  # Save visualization
                    os.path.join(
                        self.study.progressDir,
                        f"{self.study.name}-{self.study.evalCounter}",
                    ),
                )

                # Save additional metric visualizations if requested
                if getattr(self.study, "saveAdditionalMetricVisualizations", False):
                    for metricName, metricFunc in self.additionalDistanceMetrics.items():
                        metricFunc(
                            resultDomains,
                            self.study.project.targetLevelSets,
                            True,  # Save visualization
                            os.path.join(
                                self.study.progressDir,
                                f"{self.study.name}-{self.study.evalCounter}",
                            ),
                        )
        else:
            # Single-domain distance calculation (backward compatibility)
            primaryMetricStartTime = time.time()
            objectiveValue = self.distanceMetric(
                resultDomain,
                self.study.project.targetLevelSet,
                False,  # Don't save visualization yet
                os.path.join(
                    self.study.progressDir,
                    f"{self.study.name}-{self.study.evalCounter}",
                ),
            )
            primaryMetricTime = time.time() - primaryMetricStartTime

            # Store primary metric value and time for tracking
            primaryMetricName = getattr(self.study, "primaryDistanceMetric", None) or self.study.distanceMetric
            additionalMetricValues[primaryMetricName] = objectiveValue
            additionalMetricTimes[primaryMetricName] = primaryMetricTime

            # Calculate additional metrics with individual timing
            for metricName, metricFunc in self.additionalDistanceMetrics.items():
                additionalMetricStartTime = time.time()
                additionalMetricValues[metricName] = metricFunc(
                    resultDomain,
                    self.study.project.targetLevelSet,
                    False,  # Don't save visualization
                    os.path.join(
                        self.study.progressDir,
                        f"{self.study.name}-{self.study.evalCounter}",
                    ),
                )
                additionalMetricTimes[metricName] = time.time() - additionalMetricStartTime

            # Only save visualization if saveAll is True or if current evaluation is better than current best
            if saveVisualization and (
                saveAll or objectiveValue <= self.study.bestScore
            ):
                # Save primary metric visualization
                self.distanceMetric(
                    resultDomain,
                    self.study.project.targetLevelSet,
                    True,  # Save visualization
                    os.path.join(
                        self.study.progressDir,
                        f"{self.study.name}-{self.study.evalCounter}",
                    ),
                )

                # Save additional metric visualizations if requested
                if getattr(self.study, "saveAdditionalMetricVisualizations", False):
                    for metricName, metricFunc in self.additionalDistanceMetrics.items():
                        metricFunc(
                            resultDomain,
                            self.study.project.targetLevelSet,
                            True,  # Save visualization
                            os.path.join(
                                self.study.progressDir,
                                f"{self.study.name}-{self.study.evalCounter}",
                            ),
                        )

        # Track distance metric time (primary + all additional + post-processing)
        distanceMetricTime = primaryMetricTime + sum(additionalMetricTimes.values()) + postProcessingTime
        elapsedTime = time.time() - startTime

        newBest = objectiveValue <= self.study.bestScore

        if newBest:
            self.study.bestScore = objectiveValue
            self.study.bestParameters = paramDict.copy()
            self.study.bestEvaluationNumber = self.study.evalCounter

        # Save ALL evaluations to progress manager (both best and regular)
        if self.progressManager:
            # Order parameters according to metadata parameterNames
            if (
                self.progressManager.metadata
                and hasattr(self.progressManager.metadata, "parameterNames")
                and self.progressManager.metadata.parameterNames
            ):
                orderedParams = [
                    paramDict.get(name, 0.0)
                    for name in self.progressManager.metadata.parameterNames
                ]
            else:
                orderedParams = list(paramDict.values())

            self._saveEvaluationData(
                orderedParams,
                elapsedTime,
                objectiveValue,
                "progress",
                isBest=newBest,
                simulationTime=simulationTime,
                distanceMetricTime=distanceMetricTime,
                additionalMetricValues=additionalMetricValues if additionalMetricValues else None,
                additionalMetricTimes=additionalMetricTimes if additionalMetricTimes else None,
            )
        else:
            # Fallback to legacy system - only save best to progress.txt
            if newBest:
                self._saveEvaluationData(
                    list(paramDict.values()),
                    elapsedTime,
                    objectiveValue,
                    "progress",
                    isBest=True,
                    simulationTime=simulationTime,
                    distanceMetricTime=distanceMetricTime,
                    additionalMetricValues=additionalMetricValues if additionalMetricValues else None,
                    additionalMetricTimes=additionalMetricTimes if additionalMetricTimes else None,
                )

        if newBest or self.study.saveAllEvaluations:
            # Generate parameter string for filename (only for LocalSensitivityStudy)
            paramString = self._generateParameterString(paramDict, variableParamNames)
            paramSuffix = f"-{paramString}" if paramString else ""

            if self.isMultiDomainProcess:
                # Save all processed vps.Domains with corresponding names
                for domainName, processedDomain in processedDomains.items():
                    filePath = os.path.join(
                        self.study.progressDir,
                        f"{self.study.name}-{self.study.evalCounter:03d}{paramSuffix}-{domainName}.vtp",
                    )
                    # Save the processed vps.Domain (not the returned level set)
                    processedDomain.saveSurfaceMesh(filePath, True)
            else:
                # Single domain save (backward compatibility)
                filePath = os.path.join(
                    self.study.progressDir,
                    f"{self.study.name}-{self.study.evalCounter:03d}{paramSuffix}.vtp",
                )
                # Save the processed vps.Domain (not the returned level set)
                domainCopy.saveSurfaceMesh(filePath, True)

        return objectiveValue, elapsedTime, simulationTime, distanceMetricTime

    def _generateParameterString(self, paramDict: dict[str, float], variableParamNames: set[str] = None) -> str:
        """Generate a compact string representation of variable parameters for filenames."""
        if not paramDict:
            return ""

        # If no variable parameter names specified, return empty string
        if variableParamNames is None:
            return ""

        # Create abbreviated parameter names and format values for variable parameters only
        paramStrings = []
        for paramName, value in paramDict.items():
            if paramName in variableParamNames:
                # Take first 4 characters of parameter name and sanitize
                abbrevName = paramName[:4].replace('_', '').replace('-', '').lower()
                # Format value to 1 decimal place and remove unnecessary zeros
                valueStr = f"{value:.1f}".rstrip('0').rstrip('.')
                paramStrings.append(f"{abbrevName}{valueStr}")

        return "_".join(paramStrings)

    def _detectMultiDomainProcess(self) -> bool:
        """Detect if the process sequence function supports multi-domain processing."""
        if (
            not hasattr(self.study, "processSequence")
            or self.study.processSequence is None
        ):
            return False

        try:
            sig = inspect.signature(self.study.processSequence)
            params = list(sig.parameters.values())

            if len(params) >= 1:
                firstParam = params[0]
                # Check if first parameter is annotated as dict[str, Domain] or list[Domain]
                if firstParam.annotation in [dict[str, Domain], list[Domain]]:
                    return True

            return False
        except Exception:
            return False


class DlibObjectiveWrapper(BaseObjectiveWrapper):
    """Objective function wrapper for dlib optimizer."""

    def __call__(self, *x):
        """Wrapper compatible with dlib's find_min_global."""
        # Create parameter dictionary with fixed parameters
        paramDict = self.study.fixedParameters.copy()

        # Add variable parameters - x contains individual floats
        for value, (name, _) in zip(x, self.study.variableParameters.items()):
            paramDict[name] = value

        # Evaluate process
        objectiveValue, elapsedTime, simulationTime, distanceMetricTime = self._evaluateObjective(
            paramDict,
            self.study.saveVisualization,
            saveAll=self.study.saveAllEvaluations,
        )

        # Save evaluation data - only save to "all" evaluations, not duplicate with _evaluateObjective
        if not self.progressManager:
            # Only use legacy system if no progress manager
            self._saveEvaluationData(
                list(paramDict.values()),
                elapsedTime,
                objectiveValue,
                "progressAll",
                isBest=False,
                simulationTime=simulationTime,
                distanceMetricTime=distanceMetricTime,
            )

        return objectiveValue


class NevergradObjectiveWrapper(BaseObjectiveWrapper):
    """Objective function wrapper for Nevergrad optimizer."""

    def __call__(self, x):
        """
        Wrapper compatible with Nevergrad's optimization.

        Args:
            x: Array/list of parameter values (not individual arguments like dlib)
        """
        # Nevergrad passes parameters as a single array/list argument
        if hasattr(x, "__iter__") and not isinstance(x, str):
            paramValues = list(x)
        else:
            # If x is a single value, wrap it in a list
            paramValues = [x]

        # Create parameter dictionary with fixed parameters
        paramDict = self.study.fixedParameters.copy()

        # Add variable parameters - map array values to parameter names
        for value, (name, _) in zip(paramValues, self.study.variableParameters.items()):
            paramDict[name] = value

        # Evaluate process
        objectiveValue, elapsedTime, simulationTime, distanceMetricTime = self._evaluateObjective(
            paramDict,
            self.study.saveVisualization,
            saveAll=self.study.saveAllEvaluations,
        )

        # Save evaluation data - only save to "all" evaluations, not duplicate with _evaluateObjective
        if not self.progressManager:
            # Only use legacy system if no progress manager
            self._saveEvaluationData(
                list(paramDict.values()),
                elapsedTime,
                objectiveValue,
                "progressAll",
                isBest=False,
                simulationTime=simulationTime,
                distanceMetricTime=distanceMetricTime,
            )

        return objectiveValue
