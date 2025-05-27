from .fitProject import Project
from .fitObjectiveWrapper import BaseObjectiveWrapper
from .fitStudy import Study
import os
import json
from typing import Dict, List, Tuple


class LocalSensitivityStudy(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "locSensStudies")
        self.nEval = None
        # Override the progress directory name
        self.progressDir = os.path.join(self.runDir, "evaluations")
        os.makedirs(self.progressDir, exist_ok=True)

    def setVariableParameters(
        self, varParams: Dict[str, Tuple[float, float, float]], nEval: Tuple[int] = None
    ):
        """
        Set multiple parameters as variable with ranges

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
                "Parameter names must be set before defining variable parameters"
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
        """Get a dictionary of all parameter values"""
        return {name: param.value for name, param in self.parameters.items()}

    def getVariableParameterList(self):
        """Get list of variable parameters for optimization algorithms"""
        return self.variableParameters.keys()

    def getVariableBounds(self):
        """Get bounds for variable parameters as lists"""
        varParams = self.getVariableParameterList()
        lowerBounds = [p.lowerBound for p in varParams]
        upperBounds = [p.upperBound for p in varParams]
        return lowerBounds, upperBounds

    def saveParameters(self, filename: str = "parameters.json"):
        """Save parameter configuration to file"""
        filepath = os.path.join(self.runDir, filename)

        # Convert parameters to serializable dictionary
        paramDict = {
            name: {
                "value": param.value,
                "lowerBound": param.lowerBound,
                "upperBound": param.upperBound,
                "isFixed": param.isFixed,
            }
            for name, param in self.parameters.items()
        }

        with open(filepath, "w") as f:
            json.dump(paramDict, f, indent=4)

        print(f"Parameters saved to {filepath}")

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
            self.applied = True
            self.evalCounter = 0
        else:
            print("LSS has already been applied.")
            return

        try:
            # Create objective wrapper
            objectiveWrapper = BaseObjectiveWrapper(self)

            # Create base parameter dict
            baseParams = self.fixedParameters.copy()

            # Run parameter space evaluation
            results = objectiveWrapper.evaluateParameterSpace(
                baseParams, self.variableParameters, self.nEval
            )

            # Save results
            resultsPath = os.path.join(self.runDir, "sensitivity_results.json")
            with open(resultsPath, "w") as f:
                json.dump(results, f, indent=4)

            print(f"\nResults saved to: {resultsPath}")

        except Exception as e:
            print(f"LSS failed with error: {str(e)}")
            raise
