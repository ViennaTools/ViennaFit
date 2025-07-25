from typing import Dict, Tuple, Callable, List
from viennaps2d import Domain
from .fitUtilities import saveEvalToProgressFile, saveEvalToProgressManager, ProgressDataManager
from .fitDistanceMetrics import DistanceMetric
import viennals2d as vls
import time
import os


class ObjectiveWrapper:
    """Factory class for creating objective function wrappers."""

    @staticmethod
    def create(optimizer: str, optimization) -> Callable:
        """
        Create an objective function wrapper based on optimizer type.

        Args:
            optimizer: String identifying the optimizer ("dlib", etc)
            optimization: Reference to the Optimization instance

        Returns:
            Callable: Wrapped objective function compatible with chosen optimizer
        """
        if optimizer == "dlib":
            wrapper = DlibObjectiveWrapper(optimization)
        elif optimizer == "nevergrad":
            wrapper = NevergradObjectiveWrapper(optimization)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Create a regular function that calls the wrapper
        def wrappedObjectiveFunction(*args):
            return wrapper(*args)

        return wrappedObjectiveFunction


class BaseObjectiveWrapper:
    """Base class for objective function wrappers."""

    def __init__(self, study):
        self.study = study
        self.distanceMetric = DistanceMetric.create(study.distanceMetric)
        self.progressManager = getattr(study, 'progressManager', None)

    def _saveEvaluationData(
        self,
        paramValues: List[float],
        elapsedTime: float,
        objectiveValue: float,
        filename: str,
        isBest: bool = False,
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
                saveAll=True
            )
        else:
            # Fallback to legacy system
            saveEvalToProgressFile(
                [*paramValues, elapsedTime, objectiveValue],
                os.path.join(self.study.runDir, filename + ".txt"),
            )

    def _evaluateObjective(
        self, paramDict: Dict[str, float], saveVisualization: bool = False, saveAll: bool = False
    ) -> Tuple[float, float]:
        """Run process sequence and evaluate result."""
        startTime = time.time()

        # Create deep copy of initial domain
        domainCopy = Domain()
        domainCopy.deepCopy(self.study.project.initialDomain)

        # Apply process sequence
        resultDomain = self.study.processSequence(domainCopy, paramDict)

        self.study.evalCounter += 1

        # Calculate objective value using distance metric first
        objectiveValue = self.distanceMetric(
            resultDomain,
            self.study.project.targetLevelSet,
            False,  # Don't save visualization yet
            os.path.join(
                self.study.progressDir,
                f"{self.study.name}-{self.study.evalCounter}",
            ),
        )
        
        # Only save visualization if saveAll is True or if current evaluation is better than current best
        if saveVisualization and (saveAll or objectiveValue <= self.study.bestScore):
            self.distanceMetric(
                resultDomain,
                self.study.project.targetLevelSet,
                True,  # Save visualization
                os.path.join(
                    self.study.progressDir,
                    f"{self.study.name}-{self.study.evalCounter}",
                ),
            )

        elapsedTime = time.time() - startTime

        newBest = objectiveValue <= self.study.bestScore

        if newBest:
            self.study.bestScore = objectiveValue
            self.study.bestParameters = paramDict.copy()
            self.study.bestEvaluationNumber = self.study.evalCounter

        # Save ALL evaluations to progress manager (both best and regular)
        if self.progressManager:
            # Order parameters according to metadata parameterNames
            if (self.progressManager.metadata and 
                hasattr(self.progressManager.metadata, 'parameterNames') and
                self.progressManager.metadata.parameterNames):
                orderedParams = [paramDict.get(name, 0.0) for name in self.progressManager.metadata.parameterNames]
            else:
                orderedParams = list(paramDict.values())
            
            self._saveEvaluationData(
                orderedParams,
                elapsedTime,
                objectiveValue,
                "progress",
                isBest=newBest,
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
                )

        if newBest or self.study.saveAllEvaluations:
            domainCopy.saveSurfaceMesh(
                os.path.join(
                    self.study.progressDir,
                    f"{self.study.name}-{self.study.evalCounter}.vtp",
                ),
                True,
            )

        return objectiveValue, elapsedTime

    def evaluateParameterSpace(
        self,
        baseParams: Dict[str, float],
        varParams: Dict[str, Tuple[float, float, float]],
        nEvals: Tuple[int],
    ) -> Dict:
        """
        Evaluate objective function across parameter space for sensitivity analysis.

        Args:
            baseParams: Dictionary of fixed parameter values
            varParams: Dictionary mapping parameter names to (lower, poi, upper) bounds
            nEvals: Tuple of evaluation counts for each parameter

        Returns:
            Dict containing evaluation results
        """
        import numpy as np

        # Create base parameter dict with POI values for variable parameters
        evalParams = baseParams.copy()
        for paramName, (_, poi, _) in varParams.items():
            evalParams[paramName] = poi

        # First evaluate at POI
        print("\nEvaluating at Point of Interest...")
        poiValue, poiTime = self._evaluateObjective(
            evalParams, self.study.saveVisualization, saveAll=self.study.saveAllEvaluations
        )
        print(f"POI objective value: {poiValue:.6f}")

        # Then evaluate along each parameter axis
        results = {}
        for paramIdx, (paramName, (lb, poi, ub)) in enumerate(varParams.items()):
            print(f"\nEvaluating parameter: {paramName}")

            # Create parameter values to evaluate
            nPoints = nEvals[paramIdx]
            paramValues = []

            # Add points between lb and poi
            if nPoints > 1:
                leftPoints = list(np.linspace(lb, poi, (nPoints + 1) // 2)[:-1])
                paramValues.extend(leftPoints)

            # Add POI
            paramValues.append(poi)

            # Add points between poi and ub
            if nPoints > 1:
                rightPoints = list(np.linspace(poi, ub, (nPoints + 1) // 2)[1:])
                paramValues.extend(rightPoints)

            # Evaluate at each point
            paramResults = []
            for value in paramValues:
                currentParams = evalParams.copy()  # Copy from complete POI params
                currentParams[paramName] = value  # Only modify current parameter

                objValue, evalTime = self._evaluateObjective(
                    currentParams, self.study.saveVisualization, saveAll=self.study.saveAllEvaluations
                )
                # Save evaluation data
                if self.progressManager:
                    # Order parameters according to metadata parameterNames
                    if (self.progressManager.metadata and 
                        hasattr(self.progressManager.metadata, 'parameterNames') and
                        self.progressManager.metadata.parameterNames):
                        orderedParams = [currentParams.get(name, 0.0) for name in self.progressManager.metadata.parameterNames]
                    else:
                        orderedParams = list(currentParams.values())
                    
                    self._saveEvaluationData(
                        orderedParams, evalTime, objValue, "progressAll", isBest=False
                    )
                else:
                    # Fallback to legacy system
                    self._saveEvaluationData(
                        list(currentParams.values()), evalTime, objValue, "progressAll", isBest=False
                    )
                paramResults.append(
                    {"value": value, "objective": objValue, "time": evalTime}
                )
                print(f"  {paramName} = {value:.6f}: objective = {objValue:.6f}")

            results[paramName] = paramResults

        return {
            "poi": {"params": evalParams, "objective": poiValue, "time": poiTime},
            "parameter_studies": results,
        }


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
        objectiveValue, elapsedTime = self._evaluateObjective(
            paramDict, self.study.saveVisualization, saveAll=self.study.saveAllEvaluations
        )

        # Save evaluation data - only save to "all" evaluations, not duplicate with _evaluateObjective
        if not self.progressManager:
            # Only use legacy system if no progress manager
            self._saveEvaluationData(list(paramDict.values()), elapsedTime, objectiveValue, "progressAll", isBest=False)

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
        objectiveValue, elapsedTime = self._evaluateObjective(
            paramDict, self.study.saveVisualization, saveAll=self.study.saveAllEvaluations
        )

        # Save evaluation data - only save to "all" evaluations, not duplicate with _evaluateObjective
        if not self.progressManager:
            # Only use legacy system if no progress manager
            self._saveEvaluationData(
                list(paramDict.values()), elapsedTime, objectiveValue, "progressAll", isBest=False
            )

        return objectiveValue
