from .fitProject import Project
from .fitObjectiveWrapper import BaseObjectiveWrapper
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    migrateLegacyProgressFile,
    ProgressDataManager,
)
import os
import json
import numpy as np
from typing import Dict, Tuple
from datetime import datetime
from SALib.sample import saltelli
from SALib.sample import fast_sampler as fastSampler
from SALib.analyze import sobol
from SALib.analyze import fast as fastAnalyzer


class GlobalSensitivityStudy(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "globalSensStudies")
        # Override the progress directory name
        self.progressDir = os.path.join(self.runDir, "evaluations")
        os.makedirs(self.progressDir, exist_ok=True)

        # Global sensitivity specific parameters
        self.numSamples = 100  # Default value
        self.secondOrder = True  # Calculate second-order indices by default
        self.samplingMethod = "saltelli"  # Default sampling method

        # Progress management
        self.progressManager = None  # Will be initialized in apply()
        self.storageFormat = "csv"  # Default storage format

    def setVariableParameters(self, varParams: Dict[str, Tuple[float, float]]):
        """
        Set multiple parameters as variable with ranges for global sensitivity analysis.

        Args:
            varParams: Dictionary mapping parameter names to tuples of (lowerBound, upperBound)
        """
        # Check bounds validity
        for name, (lowerBound, upperBound) in varParams.items():
            if lowerBound >= upperBound:
                raise ValueError(
                    f"Invalid bounds for parameter '{name}': "
                    f"lowerBound ({lowerBound}) must be < upperBound ({upperBound})"
                )

        if not self.parameterNames:
            raise ValueError(
                "Parameter names must be set before defining variable parameters"
            )

        # Validate parameter names and store the bounds
        for name, (lowerBound, upperBound) in varParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names. "
                    f"The current parameter names are: {self.parameterNames}"
                )
            if name in self.fixedParameters:
                raise ValueError(f"Parameter '{name}' is already set as fixed")
            self.variableParameters[name] = (lowerBound, upperBound)

        return self

    def setSamplingOptions(self, numSamples: int = 100, secondOrder: bool = True):
        """
        Set options for the sensitivity analysis sampling.

        Args:
            numSamples: Base number of samples (actual sample count will be larger based on method)
            secondOrder: Whether to calculate second-order sensitivity indices
        """
        if numSamples <= 0:
            raise ValueError("numSamples must be positive")

        self.numSamples = numSamples
        self.secondOrder = secondOrder
        return self

    def setSamplingMethod(self, method: str = "saltelli"):
        """
        Set the sampling method for sensitivity analysis.

        Args:
            method: Sampling method to use. Options are:
                - 'saltelli': Saltelli's extension of Sobol sequence (default)
                - 'fast': Fourier Amplitude Sensitivity Test, more efficient for many parameters

        Returns:
            self: For method chaining
        """
        if method not in ["saltelli", "fast"]:
            raise ValueError(
                f"Invalid sampling method: {method}. "
                "Options are 'saltelli' or 'fast'."
            )

        self.samplingMethod = method
        return self

    def setStorageFormat(self, storageFormat: str):
        """Set the storage format for progress data (csv or numpy)"""
        if storageFormat.lower() not in ["csv", "numpy"]:
            raise ValueError(f"Unsupported storage format: {storageFormat}")
        self.storageFormat = storageFormat.lower()
        return self

    def _prepareSALibProblem(self):
        """Prepare the problem dictionary for SALib"""
        return {
            "num_vars": len(self.variableParameters),
            "names": list(self.variableParameters.keys()),
            "bounds": [
                self.variableParameters[name] for name in self.variableParameters.keys()
            ],
        }

    def apply(
        self,
        saveAllEvaluations: bool = False,
        saveVisualization: bool = False,
    ):
        """Apply the global sensitivity study."""
        if not self.applied:
            self.validate()
            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations
            self.applied = True
            self.evalCounter = 0

            # Initialize progress manager with metadata
            if hasattr(self, "parameterNames") and self.parameterNames:
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer="global_sensitivity",
                    createdTime=datetime.now().isoformat(),
                    description=f"Global sensitivity study for {self.name}",
                )

                # Create single progress manager (like optimization)
                progressFilepath = os.path.join(self.runDir, "progressAll")
                self.progressManager = createProgressManager(
                    progressFilepath, self.storageFormat, metadata
                )
                self.progressManager.saveMetadata()
        else:
            print("Global sensitivity study has already been applied.")
            return

        try:
            # Create objective wrapper
            objectiveWrapper = BaseObjectiveWrapper(self)

            # Create SALib problem definition
            problem = self._prepareSALibProblem()

            # Generate samples based on selected method
            if self.samplingMethod == "saltelli":
                print(
                    f"Generating {self.numSamples} base samples using Saltelli method "
                    f"(total samples: {self.numSamples*(2*len(self.variableParameters)+2)})"
                )
                paramValues = saltelli.sample(
                    problem, self.numSamples, calc_second_order=self.secondOrder
                )
            else:  # FAST method
                print(
                    f"Generating samples using FAST method "
                    f"(total samples: {self.numSamples * len(self.variableParameters)})"
                )
                paramValues = fastSampler.sample(problem, self.numSamples)

            # Save parameter samples to file
            samplesPath = os.path.join(self.runDir, "parameter_samples.csv")
            np.savetxt(
                samplesPath,
                paramValues,
                delimiter=",",
                header=",".join(problem["names"]),
                comments="",
            )
            print(f"Parameter samples saved to: {samplesPath}")

            # Evaluate model for each sample
            print(
                f"Running evaluations for {len(paramValues)} parameter combinations..."
            )
            Y = np.zeros(len(paramValues))
            resultsDetailed = []

            # Initialize tracking for best evaluations
            bestScore = float("inf")
            bestParams = None

            for i, params in enumerate(paramValues):
                # Create parameter dictionary for this evaluation
                paramDict = self.fixedParameters.copy()
                for j, name in enumerate(problem["names"]):
                    paramDict[name] = params[j]

                # Evaluate
                objectiveValue, elapsedTime = objectiveWrapper._evaluateObjective(
                    paramDict, self.saveVisualization
                )

                Y[i] = objectiveValue
                resultsDetailed.append(
                    {
                        "sampleIndex": i,
                        "parameters": {
                            name: float(value)
                            for name, value in zip(problem["names"], params)
                        },
                        "objective": float(objectiveValue),
                        "time": float(elapsedTime),
                    }
                )

                # Check if this is a new best
                if objectiveValue < bestScore:
                    bestScore = objectiveValue
                    bestParams = paramDict.copy()
                    self.bestScore = bestScore
                    self.bestParameters = bestParams
                    self.bestEvaluationNumber = i + 1

                # Progress is automatically saved by the objective wrapper
                # No need to save manually here

                # Progress reporting
                if i % 10 == 0 or i == len(paramValues) - 1:
                    print(f"  Completed {i+1}/{len(paramValues)} evaluations")
                    if bestParams:
                        print(f"    Current best score: {bestScore:.6f}")

            # Save detailed results
            resultsDetailedPath = os.path.join(self.runDir, "evaluation_results.json")
            with open(resultsDetailedPath, "w") as f:
                json.dump(resultsDetailed, f, indent=2)

            print(f"Detailed evaluation results saved to: {resultsDetailedPath}")

            # Save objective values for all samples
            np.savetxt(
                os.path.join(self.runDir, "objective_values.csv"),
                Y,
                delimiter=",",
                header="objectiveValue",
                comments="",
            )

            # Perform sensitivity analysis based on selected method
            print("Performing sensitivity analysis...")
            if self.samplingMethod == "saltelli":
                Si = sobol.analyze(
                    problem,
                    Y,
                    calc_second_order=self.secondOrder,
                    print_to_console=False,
                )

                # Format results for JSON saving
                sensitivityResults = {
                    "method": "sobol",
                    "firstOrder": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["S1"])
                    },
                    "firstOrderConf": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["S1_conf"])
                    },
                    "totalOrder": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["ST"])
                    },
                    "totalOrderConf": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["ST_conf"])
                    },
                }

                if self.secondOrder:
                    # Create second-order indices as a dict with parameter pair keys
                    secondOrderIndices = {}
                    for i, p1 in enumerate(problem["names"]):
                        for j, p2 in enumerate(problem["names"]):
                            if i > j:  # Only include unique combinations
                                secondOrderIndices[f"{p1}_{p2}"] = float(Si["S2"][i, j])

                    secondOrderConf = {}
                    for i, p1 in enumerate(problem["names"]):
                        for j, p2 in enumerate(problem["names"]):
                            if i > j:
                                secondOrderConf[f"{p1}_{p2}"] = float(
                                    Si["S2_conf"][i, j]
                                )

                    sensitivityResults["secondOrder"] = secondOrderIndices
                    sensitivityResults["secondOrderConf"] = secondOrderConf

                # Print top sensitivity indices
                print("\nParameter Sensitivities (Total Order):")
                sortedParams = sorted(
                    sensitivityResults["totalOrder"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for param, value in sortedParams:
                    conf = sensitivityResults["totalOrderConf"][param]
                    print(f"  {param}: {value:.4f} ± {conf:.4f}")

            else:  # FAST method
                Si = fastAnalyzer.analyze(problem, Y, print_to_console=False)

                # Format results for JSON saving - FAST only provides first order
                # and total order indices
                sensitivityResults = {
                    "method": "fast",
                    "firstOrder": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["S1"])
                    },
                    "firstOrderConf": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["S1_conf"])
                    },
                    "totalOrder": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["ST"])
                    },
                    "totalOrderConf": {
                        name: float(value)
                        for name, value in zip(problem["names"], Si["ST_conf"])
                    },
                }

                # Print first order indices
                print("\nParameter Sensitivities (First Order):")
                sortedParams = sorted(
                    sensitivityResults["firstOrder"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for param, value in sortedParams:
                    conf = sensitivityResults["firstOrderConf"][param]
                    print(f"  {param}: {value:.4f} ± {conf:.4f}")

            # Save sensitivity analysis results
            resultsPath = os.path.join(self.runDir, "sensitivity_results.json")
            with open(resultsPath, "w") as f:
                json.dump(sensitivityResults, f, indent=2)

            print(f"\nSensitivity analysis results saved to: {resultsPath}")

            # Print best results summary
            if bestParams:
                print(f"\nBest evaluation found:")
                print(f"  Evaluation #: {self.bestEvaluationNumber}")
                print(f"  Best score: {bestScore:.6f}")
                print("  Best parameters:")
                for name, value in bestParams.items():
                    if name in problem["names"]:  # Only show variable parameters
                        print(f"    {name}: {value:.6f}")

        except Exception as e:
            print(f"Global sensitivity study failed with error: {str(e)}")
            raise
