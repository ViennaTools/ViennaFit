from .fitProject import Project
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    saveEvalToProgressManager,
)
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime


class LocalSensitivityStudy(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "locSensStudies")
        self.nEval = None
        # Override the progress directory name
        self.progressDir = os.path.join(self.runDir, "evaluations")

        # Support for exact value evaluation mode
        self.variableParameterValues = {}  # Dict[str, List[float]] - exact values
        self.useExactValues = False  # Flag to indicate evaluation mode

        # Progress tracking
        self.progressManager = None  # Will be initialized in apply()
        self.storageFormat = "csv"  # Default storage format
        self.notes = None  # Optional notes for the sensitivity study

    def setParameterSensitivityRanges(
        self, varParams: Dict[str, Tuple[float, float, float]], nEval: Tuple[int] = None
    ):
        """
        Set parameters for sensitivity analysis with ranges around points of interest.

        Args:
            varParams: Dictionary mapping parameter names to tuples of (lowerBound, POI, upperBound)
            nEval: Tuple of integer evaluation counts per parameter
        """
        # check whether lb <= poi <= ub for each parameter
        for name, (lowerBound, poi, upperBound) in varParams.items():
            if lowerBound > poi or poi > upperBound:
                raise ValueError(
                    f"Invalid bounds for parameter '{name}': "
                    f"lowerBound ({lowerBound}) must be <= POI ({poi}) "
                    f"and POI must be <= upperBound ({upperBound})"
                )

        # if nEval is not None, check if it is a tuple of integers of the same length as varParams
        if nEval is not None:
            if not isinstance(nEval, tuple) or not all(
                isinstance(x, int) for x in nEval
            ):
                raise TypeError("nEval must be a tuple of integers")
            if len(nEval) != len(varParams):
                raise ValueError(
                    f"nEval length ({len(nEval)}) must match number of variable parameters ({len(varParams)})"
                )

        if not self.parameterNames:
            raise ValueError(
                "Parameter names must be set before defining sensitivity parameters"
            )
        for name, (lowerBound, poi, upperBound) in varParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names. "
                    f"The current parameter names are: {self.parameterNames}"
                )
            if name in self.fixedParameters:
                raise ValueError(f"Parameter '{name}' is already set as fixed")
            self.variableParameters[name] = (lowerBound, poi, upperBound)

        self.nEval = nEval if nEval is not None else (7,) * len(varParams)
        self.useExactValues = False  # Ensure range mode is set
        return self

    def setParameterGridValues(
        self, varParamValues: Dict[str, List[float]]
    ) -> "LocalSensitivityStudy":
        """
        Set parameters for grid-based evaluation with exact discrete values.

        Args:
            varParamValues: Dictionary mapping parameter names to lists of exact values to evaluate

        Example:
            study.setParameterGridValues({
                "temperature": [300, 350, 400, 450, 500],
                "pressure": [0.1, 0.5, 1.0, 2.0, 5.0]
            })
        """
        if not self.parameterNames:
            raise ValueError(
                "Parameter names must be set before defining grid parameters"
            )

        for name, valueList in varParamValues.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names. "
                    f"The current parameter names are: {self.parameterNames}"
                )
            if name in self.fixedParameters:
                raise ValueError(f"Parameter '{name}' is already set as fixed")

            if not isinstance(valueList, list) or len(valueList) == 0:
                raise ValueError(
                    f"Values for parameter '{name}' must be a non-empty list"
                )

            self.variableParameterValues[name] = valueList

        # Populate self.variableParameters for validation compatibility
        # Use min/max from values as bounds with first value as POI placeholder
        for name, values in varParamValues.items():
            minVal = min(values)
            maxVal = max(values)
            firstVal = values[0]  # Use first value as POI placeholder
            self.variableParameters[name] = (minVal, firstVal, maxVal)

        self.useExactValues = True
        print(f"Set grid values for {len(varParamValues)} parameters")
        for name, values in varParamValues.items():
            print(
                f"  {name}: {len(values)} values ({min(values):.3f} to {max(values):.3f})"
            )

        return self

    def getParameterDict(self):
        """Get a dictionary of all parameter values (fixed + variable POI values)"""
        paramDict = self.fixedParameters.copy()
        # Add POI values for variable parameters (for range-based approach)
        for name, (_, poi, _) in self.variableParameters.items():
            paramDict[name] = poi
        return paramDict

    def getVariableParameterList(self):
        """Get list of variable parameters for optimization algorithms"""
        return self.variableParameters.keys()

    def getVariableBounds(self):
        """Get bounds for variable parameters as lists"""
        lowerBounds = []
        upperBounds = []
        for name in self.variableParameters.keys():
            lowerBound, _, upperBound = self.variableParameters[name]
            lowerBounds.append(lowerBound)
            upperBounds.append(upperBound)
        return lowerBounds, upperBounds

    def setName(self, name: str):
        """Set the name for the sensitivity study (only allowed before apply() is called)"""
        if self.applied:
            raise RuntimeError("Cannot change name after study has been applied")

        # Generate new directory paths using parent class logic
        newName, newRunDir = self._generateRunDirectory(name, "locSensStudies")

        # Update name and paths (directories will be created when apply() is called)
        self.name = newName
        self.runDir = newRunDir
        self.progressDir = os.path.join(self.runDir, "evaluations")

        return self

    def getName(self) -> str:
        """Get the name of the sensitivity study"""
        return self.name

    def setNotes(self, notes: str):
        """Set notes for the sensitivity study"""
        self.notes = notes
        return self

    def saveParameters(self, filename: str = "parameters.json"):
        """Save parameter configuration to file"""
        filepath = os.path.join(self.runDir, filename)

        # Create parameter configuration dict
        paramDict = {
            "fixed": self.fixedParameters,
            "variable": self.variableParameters,
            "evaluationMode": (
                "grid_values" if self.useExactValues else "sensitivity_ranges"
            ),
        }

        if self.useExactValues:
            paramDict["gridValues"] = self.variableParameterValues

        with open(filepath, "w") as f:
            json.dump(paramDict, f, indent=4)

        print(f"Parameters saved to {filepath}")

    def _generateRangeBasedCombinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations using range-based approach (POI + variations)."""
        import numpy as np

        # Create base parameter dict with POI values for variable parameters
        baseParams = self.fixedParameters.copy()
        for paramName, (_, poi, _) in self.variableParameters.items():
            baseParams[paramName] = poi

        combinations = []

        # First add POI evaluation
        combinations.append(baseParams.copy())

        # Then add variations along each parameter axis
        for paramIdx, (paramName, (lb, poi, ub)) in enumerate(
            self.variableParameters.items()
        ):
            nPoints = self.nEval[paramIdx]
            paramValues = []

            # Add points between lb and poi
            if nPoints > 1:
                leftPoints = list(np.linspace(lb, poi, (nPoints + 1) // 2)[:-1])
                paramValues.extend(leftPoints)

            # Add POI (skip since already added)
            # paramValues.append(poi)

            # Add points between poi and ub
            if nPoints > 1:
                rightPoints = list(np.linspace(poi, ub, (nPoints + 1) // 2)[1:])
                paramValues.extend(rightPoints)

            # Create parameter combinations for this parameter
            for value in paramValues:
                paramSet = baseParams.copy()
                paramSet[paramName] = value
                combinations.append(paramSet)

        return combinations

    def _generateExactValueCombinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations using exact values approach (grid evaluation)."""
        import itertools

        # Get parameter names and their value lists
        paramNames = list(self.variableParameterValues.keys())
        paramValueLists = [self.variableParameterValues[name] for name in paramNames]

        # Generate all combinations
        combinations = []
        for combination in itertools.product(*paramValueLists):
            paramSet = self.fixedParameters.copy()
            for paramName, value in zip(paramNames, combination):
                paramSet[paramName] = value
            combinations.append(paramSet)

        return combinations

    def _evaluateSingleParameterSet(
        self, paramSet: Dict[str, float], evalNumber: int
    ) -> Dict[str, any]:
        """Evaluate a single parameter set using a fresh objective wrapper."""
        from .fitObjectiveWrapper import BaseObjectiveWrapper
        import time

        # Create fresh objective wrapper for this evaluation
        objectiveWrapper = BaseObjectiveWrapper(self)

        startTime = time.time()
        try:
            # Get variable parameter names based on evaluation mode
            if self.useExactValues:
                # Grid-based: variable parameters are those with explicit values
                variableParamNames = set(self.variableParameterValues.keys())
            else:
                # Range-based: variable parameters are those with sensitivity ranges
                variableParamNames = set(self.variableParameters.keys())

            # Single evaluation with clean state
            objValue, execTime = objectiveWrapper._evaluateObjective(
                paramSet, self.saveVisualization, saveAll=self.saveAllEvaluations,
                variableParamNames=variableParamNames
            )

            result = {
                "evaluationNumber": evalNumber,
                "parameters": paramSet,
                "objectiveValue": objValue,
                "executionTime": execTime,
                "success": True,
            }

            # Save evaluation to progress manager
            if self.progressManager:
                saveEvalToProgressManager(
                    self.progressManager,
                    evalNumber,
                    list(paramSet.values()),
                    execTime,
                    objValue,
                    saveAll=True
                )

        except Exception as e:
            result = {
                "evaluationNumber": evalNumber,
                "parameters": paramSet,
                "objectiveValue": float("inf"),
                "executionTime": time.time() - startTime,
                "success": False,
                "error": str(e),
            }

            # Save failed evaluation to progress manager
            if self.progressManager:
                saveEvalToProgressManager(
                    self.progressManager,
                    evalNumber,
                    list(paramSet.values()),
                    time.time() - startTime,
                    float("inf"),
                    saveAll=True
                )

        return result

    def validate(self):
        """Validate that all required parameters are defined (override base class)"""
        # Call base validation for common checks
        if not hasattr(self, "processSequence") or self.processSequence is None:
            raise ValueError("No process sequence has been set")

        if not self.parameterNames:
            raise ValueError("No parameters have been defined")

        if not self.distanceMetric:
            raise ValueError("No distance metric has been set")

        # Check that parameters are properly configured for the selected mode
        if self.useExactValues:
            # Grid values mode validation
            if not self.variableParameterValues:
                raise ValueError(
                    "No grid parameter values have been set. Use setParameterGridValues()."
                )
        else:
            # Sensitivity ranges mode validation
            if not self.variableParameters:
                raise ValueError(
                    "No sensitivity parameter ranges have been set. Use setParameterSensitivityRanges()."
                )

        # Standard validation: union of fixed and variable parameters must match parameter names
        if set(self.fixedParameters.keys()).union(
            set(self.variableParameters.keys())
        ) != set(self.parameterNames):
            raise ValueError(
                "The union of fixed and variable parameters does not match the parameter names"
            )

        return True

    def apply(
        self,
        saveAllEvaluations: bool = True,
        saveVisualization: bool = True,
    ):
        """Apply the sensitivity study."""
        if not self.applied:
            self.validate()
            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations

            # Create directories (similar to fitOptimization.py)
            os.makedirs(self.runDir, exist_ok=True)
            os.makedirs(self.progressDir, exist_ok=True)

            self.applied = True
            self.evalCounter = 0

            # Initialize progress manager with metadata
            if hasattr(self, "parameterNames") and self.parameterNames:
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer="local_sensitivity",
                    createdTime=datetime.now().isoformat(),
                    description=f"Local sensitivity study for {self.name}",
                )

                progressFilepath = os.path.join(self.runDir, "progressAll")
                self.progressManager = createProgressManager(
                    progressFilepath, self.storageFormat, metadata
                )
                self.progressManager.saveMetadata()

            # Save notes to file if provided
            if self.notes is not None:
                notesFile = os.path.join(self.runDir, "notes.txt")
                with open(notesFile, "w") as f:
                    f.write(self.notes)
                print(f"Notes saved to {notesFile}")
        else:
            print("LSS has already been applied.")
            return

        try:
            # Determine evaluation mode and generate parameter combinations
            if self.useExactValues:
                print("Running sensitivity study with grid parameter values...")
                paramCombinations = self._generateExactValueCombinations()
                evaluationMode = "grid_values"
            else:
                print("Running sensitivity study with sensitivity range exploration...")
                paramCombinations = self._generateRangeBasedCombinations()
                evaluationMode = "sensitivity_ranges"

            print(f"Total parameter combinations to evaluate: {len(paramCombinations)}")

            # Evaluate all parameter combinations
            results = []
            for i, paramSet in enumerate(paramCombinations, 1):
                print(f"Evaluation {i}/{len(paramCombinations)}: {paramSet}")

                result = self._evaluateSingleParameterSet(paramSet, i)
                results.append(result)

                if result["success"]:
                    print(f"  Objective value: {result['objectiveValue']:.6f}")
                    print(f"  Execution time: {result['executionTime']:.2f} seconds")
                else:
                    print(f"  Failed: {result.get('error', 'Unknown error')}")

            # Create comprehensive results dictionary
            successfulResults = [r for r in results if r["success"]]
            failedResults = [r for r in results if not r["success"]]

            finalResults = {
                "metadata": {
                    "studyName": self.name,
                    "evaluationMode": evaluationMode,
                    "totalEvaluations": len(results),
                    "successfulEvaluations": len(successfulResults),
                    "failedEvaluations": len(failedResults),
                },
                "configuration": {
                    "fixedParameters": self.fixedParameters,
                    "variableParameters": (
                        self.variableParameters if not self.useExactValues else {}
                    ),
                    "variableParameterValues": (
                        self.variableParameterValues if self.useExactValues else {}
                    ),
                    "nEval": self.nEval if not self.useExactValues else None,
                },
                "evaluations": results,
                "summary": {
                    "bestObjectiveValue": (
                        min(r["objectiveValue"] for r in successfulResults)
                        if successfulResults
                        else None
                    ),
                    "worstObjectiveValue": (
                        max(r["objectiveValue"] for r in successfulResults)
                        if successfulResults
                        else None
                    ),
                    "averageObjectiveValue": (
                        sum(r["objectiveValue"] for r in successfulResults)
                        / len(successfulResults)
                        if successfulResults
                        else None
                    ),
                    "averageExecutionTime": (
                        sum(r["executionTime"] for r in successfulResults)
                        / len(successfulResults)
                        if successfulResults
                        else None
                    ),
                },
            }

            # Save results
            resultsPath = os.path.join(self.runDir, "sensitivity_results.json")
            with open(resultsPath, "w") as f:
                json.dump(finalResults, f, indent=4)

            print(f"\nSensitivity study completed!")
            print(f"Results saved to: {resultsPath}")
            print(f"Successful evaluations: {len(successfulResults)}/{len(results)}")
            if successfulResults:
                print(
                    f"Best objective value: {min(r['objectiveValue'] for r in successfulResults):.6f}"
                )

        except Exception as e:
            print(f"LSS failed with error: {str(e)}")
            raise
