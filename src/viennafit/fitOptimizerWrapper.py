from typing import Dict, List, Tuple, Any
from .fitObjectiveWrapper import ObjectiveWrapper


class OptimizerWrapper:
    """Factory class for creating optimizer wrappers."""

    @staticmethod
    def create(optimizer: str, optimization) -> "BaseOptimizerWrapper":
        """
        Create an optimizer wrapper based on optimizer type.

        Args:
            optimizer: String identifying the optimizer ("dlib", "nevergrad", "ax", "botorch")
            optimization: Reference to the Optimization instance

        Returns:
            BaseOptimizerWrapper: Appropriate optimizer wrapper
        """
        if optimizer == "dlib":
            return DlibOptimizerWrapper(optimization)
        elif optimizer == "nevergrad":
            return NevergradOptimizerWrapper(optimization)
        elif optimizer in ["ax", "botorch"]:
            return AxOptimizerWrapper(optimization)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")


class BaseOptimizerWrapper:
    """Base class for optimizer wrappers."""

    def __init__(self, optimization):
        self._optimization = optimization

    def getBounds(self) -> Tuple[List[float], List[float]]:
        """Get parameter bounds as separate lower and upper bound lists."""
        lowerBounds = []
        upperBounds = []
        for lower, upper in self._optimization.variableParameters.values():
            lowerBounds.append(lower)
            upperBounds.append(upper)

        # Verify bounds
        if not lowerBounds or not upperBounds:
            raise ValueError("No bounds defined for variable parameters")
        if len(lowerBounds) != len(self._optimization.variableParameters):
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
        from .fitExceptions import EarlyStoppingException

        # Get bounds
        lowerBounds, upperBounds = self.getBounds()

        # Create objective function
        objectiveFunction = ObjectiveWrapper.create("dlib", self._optimization)

        # Run optimization
        earlyStopped = False
        try:
            x, fx = find_min_global(
                objectiveFunction,
                lowerBounds,
                upperBounds,
                numEvaluations,
            )
        except EarlyStoppingException:
            earlyStopped = True
            # Use best parameters found so far
            parameterNames = list(self._optimization.variableParameters.keys())
            x = [self._optimization.bestParameters.get(name) for name in parameterNames]
            fx = self._optimization.bestScore

        # Format results
        parameterNames = list(self._optimization.variableParameters.keys())
        optimizedParams = dict(zip(parameterNames, x))

        return {
            "success": True,
            "x": optimizedParams,
            "fun": fx,
            "nfev": self._optimization._evalCounter,
            "earlyStopped": earlyStopped,
        }


class NevergradOptimizerWrapper(BaseOptimizerWrapper):
    """Wrapper for Nevergrad optimizer."""

    def optimize(self, numEvaluations: int) -> Dict[str, Any]:
        """Run optimization using Nevergrad."""
        import nevergrad as ng
        from .fitExceptions import EarlyStoppingException

        # Get bounds and parameter names in consistent order
        parameterNames = list(self._optimization.variableParameters.keys())
        lowerBounds, upperBounds = self.getBounds()

        # Create starting point at center of bounds
        startingPoint = [
            (lower + upper) / 2 for lower, upper in zip(lowerBounds, upperBounds)
        ]

        # Create parametrization
        parametrization = ng.p.Array(init=startingPoint)
        parametrization.set_bounds(lowerBounds, upperBounds)

        # Create objective function
        objectiveFunction = ObjectiveWrapper.create("nevergrad", self._optimization)

        # Create optimizer
        optimizer = ng.optimizers.NGOpt4(
            parametrization=parametrization,
            budget=numEvaluations,
            num_workers=1,  # Single-threaded for simplicity
        )

        # Run optimization
        earlyStopped = False
        try:
            recommendation = optimizer.minimize(objectiveFunction)
            optimizedParamValues = recommendation.value
            bestLoss = recommendation.loss
        except EarlyStoppingException:
            earlyStopped = True
            # Get best so far from recommendation
            recommendation = optimizer.recommend()
            optimizedParamValues = recommendation.value
            bestLoss = self._optimization.bestScore

        # Format results - recommendation.value is an array, map to parameter names
        optimizedParams = dict(zip(parameterNames, optimizedParamValues))

        return {
            "success": True,
            "x": optimizedParams,
            "fun": bestLoss,
            "nfev": self._optimization._evalCounter,
            "earlyStopped": earlyStopped,
        }


class AxOptimizerWrapper(BaseOptimizerWrapper):
    """Wrapper for Ax/BoTorch optimizer using qExpectedImprovement."""

    def _shouldStopEarly(self) -> bool:
        """Check if early stopping criterion is met."""
        if self._optimization.earlyStoppingPatience is None:
            return False
        if (
            self._optimization._evalCounter
            < self._optimization.earlyStoppingMinEvaluations
        ):
            return False
        return (
            self._optimization._evaluationsSinceImprovement
            >= self._optimization.earlyStoppingPatience
        )

    def optimize(
        self, numEvaluations: int
    ) -> Dict[str, Any]:  # numEvaluations is validated in apply(), unused here
        """Run Bayesian optimization using Ax with BoTorch qEI acquisition function."""
        from ax import Client, RangeParameterConfig
        from .fitExceptions import EarlyStoppingException

        # Get configuration
        parameterNames = list(self._optimization.variableParameters.keys())
        lowerBounds, upperBounds = self.getBounds()

        # Get batch configuration (already validated in apply())
        batchSize = getattr(self._optimization, "batchSize", 4)
        initialSamples = getattr(
            self._optimization, "initialSamples", max(5, 2 * len(parameterNames))
        )
        numBatches = (
            self._optimization.numBatches
        )  # Guaranteed to be set by apply() validation

        # Calculate total evaluations
        totalEvaluations = initialSamples + (numBatches * batchSize)

        print(f"Ax/BoTorch configuration:")
        print(f"  Initial samples (Sobol): {initialSamples}")
        print(f"  Batch size: {batchSize}")
        print(f"  Number of batches: {numBatches}")
        print(
            f"  Total evaluations: {totalEvaluations} = {initialSamples} + ({numBatches} × {batchSize})"
        )

        # Get primary metric name for objective
        primaryMetricName = (
            getattr(self._optimization, "_primaryDistanceMetric", None)
            or self._optimization.distanceMetric
        )

        # Create Ax client
        axClient = Client(random_seed=None)

        # Configure experiment with parameter space
        parameters = []
        for paramName, (lower, upper) in zip(
            parameterNames, zip(lowerBounds, upperBounds)
        ):
            parameters.append(
                RangeParameterConfig(
                    name=paramName,
                    parameter_type="float",
                    bounds=(float(lower), float(upper)),
                )
            )

        axClient.configure_experiment(
            parameters=parameters,
            name=self._optimization.name,
        )

        # Configure optimization objective (prefix with '-' for minimization)
        axClient.configure_optimization(
            objective=f"-{primaryMetricName}",
        )

        # Configure generation strategy: Sobol initialization + BoTorch
        axClient.configure_generation_strategy(
            method="quality",  # Use quality (BoTorch) method
            initialization_budget=initialSamples,
        )

        # Get objective wrapper
        objectiveWrapper = ObjectiveWrapper.create("ax", self._optimization)

        # Run optimization loop
        numTrials = 0
        earlyStopped = False
        while numTrials < totalEvaluations:
            # Check early stopping BEFORE generating new trials
            if self._shouldStopEarly():
                self._optimization.earlyStoppedAt = self._optimization._evalCounter
                earlyStopped = True
                break

            # Generate batch of trials
            trialsToEvaluate = []
            for _ in range(min(batchSize, totalEvaluations - numTrials)):
                try:
                    trial = axClient.get_next_trials(max_trials=1)
                    if trial:
                        trialsToEvaluate.extend(trial.items())
                except Exception as e:
                    # No more trials to generate
                    print(f"No more trials: {e}")
                    break

            if not trialsToEvaluate:
                break

            # Evaluate batch
            for trialIndex, parameterization in trialsToEvaluate:
                # Check early stopping before each evaluation
                if self._shouldStopEarly():
                    self._optimization.earlyStoppedAt = self._optimization._evalCounter
                    earlyStopped = True
                    break

                # Evaluate single trial
                try:
                    result = objectiveWrapper.evaluateTrial(parameterization)
                except EarlyStoppingException:
                    earlyStopped = True
                    break

                # Complete trial with result
                axClient.complete_trial(trial_index=trialIndex, raw_data=result)
                numTrials += 1

            if earlyStopped:
                break

        # Get best parameters
        # Returns: (parameters_dict, metrics_dict, trial_index, arm_name)
        bestResult = axClient.get_best_parameterization()
        print(f"\nAx best result structure: {bestResult}")

        bestParameters = bestResult[0]
        metricsDict = bestResult[1]

        print(f"Best parameters: {bestParameters}")
        print(f"Metrics dict: {metricsDict}")
        print(f"Primary metric name: {primaryMetricName}")

        # Extract objective value
        bestObjective = float("inf")

        # The metric is stored with the plain metric name
        print(f"Looking for metric key: '{primaryMetricName}'")

        if primaryMetricName in metricsDict:
            metricValue = metricsDict[primaryMetricName]
            print(f"Found metric with key '{primaryMetricName}': {metricValue}")
            # Check if it's a tuple (mean, sem) or just a number
            if isinstance(metricValue, tuple):
                # Extract the mean value (no negation needed - Ax handles minimization)
                bestObjective = metricValue[0]
                print(f"Extracted value: {bestObjective}")
            else:
                bestObjective = metricValue
                print(f"Extracted value (non-tuple): {bestObjective}")
        else:
            print(f"ERROR: Could not find metric '{primaryMetricName}' in results.")
            print(f"Available metrics: {list(metricsDict.keys())}")
            print(f"Full metrics dict: {metricsDict}")

        return {
            "success": True,
            "x": bestParameters,
            "fun": bestObjective,
            "nfev": numTrials,
            "earlyStopped": earlyStopped,
        }
