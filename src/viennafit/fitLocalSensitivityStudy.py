from .fitProject import Project
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    saveEvalToProgressManager,
    getViennaVersionInfo,
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
        self._progressDir = os.path.join(self.runDir, "evaluations")

        # Progress tracking
        self._progressManager = None  # Will be initialized in apply()
        self._storageFormat = "csv"  # Default storage format
        self.notes = None  # Optional notes for the sensitivity study
        self._bestScore = float("inf")

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
        if self._applied:
            raise RuntimeError("Cannot change name after study has been applied")

        # Generate new directory paths using parent class logic
        newName, newRunDir = self._generateRunDirectory(name, "locSensStudies")

        # Update name and paths (directories will be created when apply() is called)
        self.name = newName
        self.runDir = newRunDir
        self._progressDir = os.path.join(self.runDir, "evaluations")

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

        paramDict = {
            "fixed": self.fixedParameters,
            "variable": self.variableParameters,
        }

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
            variableParamNames = set(self.variableParameters.keys())

            # Single evaluation with clean state
            objValue, execTime, _, _ = objectiveWrapper._evaluateObjective(
                paramSet,
                self.saveComparison,
                saveAll=self.saveAllEvaluations,
                variableParamNames=variableParamNames,
            )

            isBest = objValue < self._bestScore
            if isBest:
                self._bestScore = objValue

            result = {
                "evaluationNumber": evalNumber,
                "parameters": paramSet,
                "objectiveValue": objValue,
                "executionTime": execTime,
                "success": True,
            }

            # Save evaluation to progress manager
            if self._progressManager:
                saveEvalToProgressManager(
                    self._progressManager,
                    evalNumber,
                    list(paramSet.values()),
                    execTime,
                    objValue,
                    isBest=isBest,
                    saveAll=True,
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
            if self._progressManager:
                saveEvalToProgressManager(
                    self._progressManager,
                    evalNumber,
                    list(paramSet.values()),
                    time.time() - startTime,
                    float("inf"),
                    isBest=False,
                    saveAll=True,
                )

        return result

    def validate(self):
        """Validate that all required parameters are defined (override base class)"""
        if not hasattr(self, "processSequence") or self.processSequence is None:
            raise ValueError("No process sequence has been set")

        if not self.parameterNames:
            raise ValueError("No parameters have been defined")

        if not self.distanceMetric:
            raise ValueError("No distance metric has been set")

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
        saveComparison: bool = True,
    ):
        """Apply the sensitivity study.

        Runs a one-at-a-time (OAT) local sensitivity analysis over the ranges defined
        via setParameterSensitivityRanges(). Multi-domain projects are supported
        automatically through the underlying BaseObjectiveWrapper.
        """
        if not self._applied:
            self.validate()
            self.saveComparison = saveComparison
            self.saveAllEvaluations = saveAllEvaluations

            # Create directories (similar to fitOptimization.py)
            os.makedirs(self.runDir, exist_ok=True)
            os.makedirs(self._progressDir, exist_ok=True)

            self._applied = True
            self._evalCounter = 0

            # Initialize progress manager with metadata
            if hasattr(self, "parameterNames") and self.parameterNames:
                versionInfo = getViennaVersionInfo()
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer="local_sensitivity",
                    createdTime=datetime.now().isoformat(),
                    description=f"Local sensitivity study for {self.name}",
                    notes=self.notes,
                    viennapsVersion=versionInfo["viennapsVersion"],
                    viennalsVersion=versionInfo["viennalsVersion"],
                    viennapsCommit=versionInfo["viennapsCommit"],
                    viennalsCommit=versionInfo["viennalsCommit"],
                )

                progressFilepath = os.path.join(self.runDir, "progressAll")
                self._progressManager = createProgressManager(
                    progressFilepath, self._storageFormat, metadata
                )
                self._progressManager.saveMetadata()

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
            print("Running sensitivity study with sensitivity range exploration...")
            paramCombinations = self._generateRangeBasedCombinations()

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
                    "totalEvaluations": len(results),
                    "successfulEvaluations": len(successfulResults),
                    "failedEvaluations": len(failedResults),
                },
                "configuration": {
                    "fixedParameters": self.fixedParameters,
                    "variableParameters": self.variableParameters,
                    "nEval": self.nEval,
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
