from typing import Dict, List, Tuple, Any
from .fitObjectiveWrapper import ObjectiveWrapper


class OptimizerWrapper:
    """Factory class for creating optimizer wrappers."""

    @staticmethod
    def create(optimizer: str, optimization) -> "BaseOptimizerWrapper":
        """
        Create an optimizer wrapper based on optimizer type.

        Args:
            optimizer: String identifying the optimizer ("dlib", etc)
            optimization: Reference to the Optimization instance

        Returns:
            BaseOptimizerWrapper: Appropriate optimizer wrapper
        """
        if optimizer == "dlib":
            return DlibOptimizerWrapper(optimization)
        elif optimizer == "nevergrad":
            return NevergradOptimizerWrapper(optimization)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")


class BaseOptimizerWrapper:
    """Base class for optimizer wrappers."""

    def __init__(self, optimization):
        self.optimization = optimization

    def getBounds(self) -> Tuple[List[float], List[float]]:
        """Get parameter bounds as separate lower and upper bound lists."""
        lowerBounds = []
        upperBounds = []
        for lower, upper in self.optimization.variableParameters.values():
            lowerBounds.append(lower)
            upperBounds.append(upper)

        # Verify bounds
        if not lowerBounds or not upperBounds:
            raise ValueError("No bounds defined for variable parameters")
        if len(lowerBounds) != len(self.optimization.variableParameters):
            raise ValueError("Missing bounds for some variable parameters")

        return lowerBounds, upperBounds

    def optimize(self, numEvaluations: int) -> Dict[str, Any]:
        """
        Run the optimization.

        Args:
            numEvaluations: Maximum number of function evaluations

        Returns:
            Dict containing optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize()")


class DlibOptimizerWrapper(BaseOptimizerWrapper):
    """Wrapper for dlib optimizer."""

    def optimize(self, numEvaluations: int) -> Dict[str, Any]:
        """Run optimization using dlib's find_min_global."""
        from dlib import find_min_global

        # Get bounds
        lowerBounds, upperBounds = self.getBounds()

        # Create objective function
        objectiveFunction = ObjectiveWrapper.create("dlib", self.optimization)

        # Run optimization
        x, fx = find_min_global(
            objectiveFunction,
            lowerBounds,
            upperBounds,
            numEvaluations,
        )

        # Format results
        parameterNames = list(self.optimization.variableParameters.keys())
        optimizedParams = dict(zip(parameterNames, x))

        return {
            "success": True,
            "x": optimizedParams,
            "fun": fx,
            "nfev": numEvaluations,
        }


class NevergradOptimizerWrapper(BaseOptimizerWrapper):
    """Wrapper for Nevergrad optimizer."""

    def optimize(self, numEvaluations: int) -> Dict[str, Any]:
        """Run optimization using Nevergrad."""
        import nevergrad as ng

        # Get bounds and parameter names in consistent order
        parameterNames = list(self.optimization.variableParameters.keys())
        lowerBounds, upperBounds = self.getBounds()

        # Create starting point at center of bounds
        startingPoint = [
            (lower + upper) / 2 for lower, upper in zip(lowerBounds, upperBounds)
        ]

        # Create parametrization
        parametrization = ng.p.Array(init=startingPoint)
        parametrization.set_bounds(lowerBounds, upperBounds)

        # Create objective function
        objectiveFunction = ObjectiveWrapper.create("nevergrad", self.optimization)

        # Create optimizer
        optimizer = ng.optimizers.NGOpt4(
            parametrization=parametrization,
            budget=numEvaluations,
            num_workers=1,  # Single-threaded for simplicity
        )

        # Run optimization
        recommendation = optimizer.minimize(objectiveFunction)

        # Format results - recommendation.value is an array, map to parameter names
        optimizedParamValues = recommendation.value
        optimizedParams = dict(zip(parameterNames, optimizedParamValues))

        return {
            "success": True,
            "x": optimizedParams,
            "fun": recommendation.loss,
            "nfev": numEvaluations,
        }
