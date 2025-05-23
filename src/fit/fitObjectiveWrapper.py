from typing import Dict, Tuple, Callable, List
from viennaps2d import Domain
from .fitUtilities import saveEvalToProgressFile
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

            # Create a regular function that calls the wrapper
            def wrappedObjectiveFunction(*args):
                return wrapper(*args)

            return wrappedObjectiveFunction
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")


class BaseObjectiveWrapper:
    """Base class for objective function wrappers."""

    def __init__(self, optimization):
        self.optimization = optimization
        self.distanceMetric = DistanceMetric.create(optimization.distanceMetric)

    def _saveEvaluationData(
        self, paramValues: List[float], elapsedTime: float, objectiveValue: float
    ):
        """Save evaluation data to progress file."""
        saveEvalToProgressFile(
            [*paramValues, elapsedTime, objectiveValue],
            os.path.join(self.optimization.runDir, "progress.txt"),
        )

    def _evaluateObjective(
        self, paramDict: Dict[str, float], saveVisualization: bool = False
    ) -> Tuple[float, float]:
        """Run process sequence and evaluate result."""
        startTime = time.time()

        # Create deep copy of initial domain
        domainCopy = Domain()
        domainCopy.deepCopy(self.optimization.project.initialDomain)

        # Apply process sequence
        resultDomain = self.optimization.processSequence(domainCopy, paramDict)

        self.optimization.evalCounter += 1

        # Calculate objective value using distance metric
        objectiveValue = self.distanceMetric(
            resultDomain,
            self.optimization.project.targetLevelSet,
            saveVisualization,
            os.path.join(
                self.optimization.progressDir,
                f"{self.optimization.name}-{self.optimization.evalCounter}",
            ),
        )

        elapsedTime = time.time() - startTime

        newBest = objectiveValue <= self.optimization.bestScore

        if newBest:
            self.optimization.bestScore = objectiveValue
            self.optimization.bestParameters = paramDict.copy()

        if newBest or self.optimization.saveAllEvaluations:
            domainCopy.saveSurfaceMesh(
                os.path.join(
                    self.optimization.progressDir,
                    f"{self.optimization.name}-{self.optimization.evalCounter}.vtp",
                ),
                True,
            )

        return objectiveValue, elapsedTime


class DlibObjectiveWrapper(BaseObjectiveWrapper):
    """Objective function wrapper for dlib optimizer."""

    def __call__(self, *x):
        """Wrapper compatible with dlib's find_min_global."""
        # Create parameter dictionary with fixed parameters
        paramDict = self.optimization.fixedParameters.copy()

        # Add variable parameters - x contains individual floats
        for value, (name, _) in zip(x, self.optimization.variableParameters.items()):
            paramDict[name] = value

        # Evaluate process
        objectiveValue, elapsedTime = self._evaluateObjective(
            paramDict, self.optimization.saveVisualization
        )

        # Save evaluation data
        self._saveEvaluationData(list(x), elapsedTime, objectiveValue)

        return objectiveValue
