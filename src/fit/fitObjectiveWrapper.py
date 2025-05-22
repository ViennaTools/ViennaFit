from typing import Dict, Tuple, Callable
from viennaps2d import Domain
from .fitUtilities import saveEvalToProgressFile
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

    def _evaluateProcess(self, paramDict: Dict[str, float]) -> Tuple[float, float]:
        """
        Evaluate process sequence with given parameters.

        Returns:
            Tuple[float, float]: (objective_value, elapsed_time)
        """
        startTime = time.time()

        # Create deep copy of initial domain
        domainCopy = Domain()
        domainCopy.deepCopy(self.optimization.project.initialDomain)

        # Apply process sequence
        resultDomain = self.optimization.processSequence(domainCopy, paramDict)

        # Calculate objective value based on distance metric
        if self.optimization.distanceMetric == "CA+CSF":
            ca = vls.CompareArea(resultDomain, self.optimization.project.targetLevelSet)
            ca.apply()
            csf = vls.CompareSparseField(
                resultDomain, self.optimization.project.targetLevelSet
            )
            csf.apply()
            objectiveValue = ca.getAreaMismatch() + csf.getSumSquaredDifferences()
        else:
            raise NotImplementedError(
                f"Distance metric {self.optimization.distanceMetric} not implemented"
            )

        elapsedTime = time.time() - startTime

        return objectiveValue, elapsedTime


class DlibObjectiveWrapper(BaseObjectiveWrapper):
    """Objective function wrapper for dlib optimizer."""

    def __call__(self, *x):
        """
        Wrapper compatible with dlib's find_min_global.

        Args:
            *x: Variable parameter values in order of definition

        Returns:
            float: Objective function value
        """
        # Create parameter dictionary with fixed parameters
        paramDict = self.optimization.fixedParameters.copy()

        # Add variable parameters - x contains individual floats
        for value, (name, _) in zip(x, self.optimization.variableParameters.items()):
            paramDict[name] = value

        # Evaluate process
        objectiveValue, elapsedTime = self._evaluateProcess(paramDict)

        # Update best result if better
        if objectiveValue < self.optimization.bestScore:
            self.optimization.bestScore = objectiveValue
            self.optimization.bestParameters = paramDict.copy()

        # Save evaluation data
        saveEvalToProgressFile(
            [*x, elapsedTime, objectiveValue],
            os.path.join(self.optimization.runDir, "progress.txt"),
        )

        return objectiveValue
