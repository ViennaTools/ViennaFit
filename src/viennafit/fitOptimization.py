from .fitProject import Project
from .fitOptimizerWrapper import OptimizerWrapper
from .fitStudy import Study
import viennaps2d as vps
import os
import json
import shutil
from typing import Dict, List, Tuple


class Optimization(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "optimizationRuns")
        self.optimizer = "dlib"  # Default optimizer

    def setParameterNames(self, paramNames: List[str]):
        """Specifies names of parameters that will be used in optimization"""
        self.parameterNames = paramNames
        return self

    def setFixedParameters(self, fixedParams: Dict[str, float]):
        """
        Set multiple parameters as fixed with specific values

        Args:
            fixedParams: Dictionary mapping parameter names to fixed values
        """
        if self.parameterNames is None:
            raise ValueError(
                "Parameter names must be set before defining fixed parameters"
            )
        for name, value in fixedParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names"
                )
            if name in self.variableParameters:
                raise ValueError(f"Parameter '{name}' is already set as variable")
            self.fixedParameters[name] = value
        return self

    def setVariableParameters(self, varParams: Dict[str, Tuple[float, float]]):
        """
        Set multiple parameters as variable with ranges

        Args:
            varParams: Dictionary mapping parameter names to tuples of (lowerBound, upperBound)
        """
        if not self.parameterNames:
            raise ValueError(
                "Parameter names must be set before defining variable parameters"
            )
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

    def getVariableParameterList(self):
        """Get list of variable parameters for optimization algorithms"""
        return self.variableParameters.keys()

    def getVariableBounds(self):
        """Get bounds for variable parameters as lists"""
        varParams = self.getVariableParameterList()
        lowerBounds = [p.lowerBound for p in varParams]
        upperBounds = [p.upperBound for p in varParams]
        return lowerBounds, upperBounds

    def saveResults(self, filename: str = "results.json"):
        """Save results to file"""
        filepath = os.path.join(self.runDir, filename)

        result = {
            "bestScore": self.bestScore,
            "bestEvaluation#": self.bestEvaluationNumber,
            "bestParameters": self.bestParameters,
            "fixedParameters": self.fixedParameters,
            "variableParameters": self.variableParameters,
            "optimizer": self.optimizer,
            "numEvaluations": self.numEvaluations,
        }

        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)

        bestDomainFile = f"{self.name}-{self.bestEvaluationNumber}.vtp"
        bestDomainPath = os.path.join(self.runDir, "progress", bestDomainFile)
        if os.path.exists(bestDomainPath):
            projectDomainDir = os.path.join(
                self.project.projectAbsPath, "domains", "optimalDomains"
            )
            os.makedirs(projectDomainDir, exist_ok=True)
            targetPath = os.path.join(projectDomainDir, bestDomainFile)
            shutil.copy2(bestDomainPath, targetPath)
            print(f"Best domain copied to {targetPath}")
        else:
            print(f"Best domain file {bestDomainFile} not found in progress directory")

        print(f"Results saved to {filepath}")

    def saveStartingConfiguration(self):
        """Save the starting configuration of the optimization"""
        if not self.applied:
            raise RuntimeError(
                "Optimization must be applied before saving configuration"
            )

        config = {
            "name": self.name,
            "parameterNames": self.parameterNames,
            "fixedParameters": self.fixedParameters,
            "variableParameters": self.variableParameters,
            "optimizer": self.optimizer,
            "numEvaluations": self.numEvaluations,
        }

        configFile = os.path.join(
            self.runDir, self.name + "-startingConfiguration.json"
        )
        with open(configFile, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Starting configuration saved to {configFile}")

    def setOptimizer(self, optimizer: str):
        """Set the optimizer to be used"""
        self.optimizer = optimizer
        return self

    def apply(
        self,
        numEvaluations: int = 100,
        saveAllEvaluations: bool = False,
        saveVisualization: bool = True,
    ):
        """Apply the optimization."""
        if not self.applied:
            self.validate()
            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations
            self.applied = True
            self.evalCounter = 0
            self.numEvaluations = numEvaluations
            self.saveStartingConfiguration()
        else:
            print("Optimization has already been applied.")
            return

        # Create optimizer wrapper
        optimizer = OptimizerWrapper.create(self.optimizer, self)

        try:
            # Run optimization
            result = optimizer.optimize(self.numEvaluations)

            # Save results
            if result["success"]:
                if self.bestParameters is None:
                    self.bestParameters = {}
                self.bestParameters.update(result["x"])
                self.bestScore = result["fun"]

                print(f"Optimization completed successfully:")
                print(f"  Function evaluations: {result['nfev']}")
                print(f"  Best score: {result['fun']:.6f}")
                print("  Best parameters:")
                for name, value in result["x"].items():
                    print(f"    {name}: {value:.6f}")
                print(f" Best evaluation #: {self.bestEvaluationNumber}")

                # Save final results
                self.saveResults(self.name + "-final-results.json")

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise
