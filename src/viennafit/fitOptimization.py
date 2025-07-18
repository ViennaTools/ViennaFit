from .fitProject import Project
from .fitOptimizerWrapper import OptimizerWrapper
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    migrateLegacyProgressFile,
    ProgressDataManager,
)
import viennaps2d as vps
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
import ast
from typing import Dict, List, Tuple
from datetime import datetime


class Optimization(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "optimizationRuns")
        self.optimizer = "dlib"  # Default optimizer
        self.progressManager = None  # Will be initialized in apply()
        self.storageFormat = "csv"  # Default storage format

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

    def _parseProgressFile(self, file_path: str):
        """Parse progress file containing Python list strings into numpy array"""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            data_rows = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    try:
                        # Parse the Python list string
                        data_list = ast.literal_eval(line)
                        if isinstance(data_list, list):
                            # Convert strings to floats
                            data_row = [float(x) for x in data_list]
                            data_rows.append(data_row)
                    except (ValueError, SyntaxError):
                        # Skip malformed lines
                        continue

            if data_rows:
                # Convert to numpy array, handling variable column counts
                max_cols = max(len(row) for row in data_rows)
                # Pad shorter rows with NaN if needed
                padded_rows = []
                for row in data_rows:
                    if len(row) < max_cols:
                        row.extend([np.nan] * (max_cols - len(row)))
                    padded_rows.append(row)

                return np.array(padded_rows)
            else:
                return np.array([])

        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return np.array([])

    def saveVisualizationPlots(self):
        """Save matplotlib visualization plots after optimization completion"""
        if not self.applied or not self.bestParameters:
            print("No optimization results to visualize")
            return

        # Create plots directory
        plots_dir = os.path.join(self.runDir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Variable parameter position plots
        if self.variableParameters:
            # Create subplots with each parameter having its own y-axis
            num_params = len(self.variableParameters)
            fig, axes = plt.subplots(
                num_params, 1, figsize=(8, max(2.5, num_params * 1.2)), squeeze=False
            )

            # Adjust spacing between subplots
            plt.subplots_adjust(hspace=0.4)

            param_names = list(self.variableParameters.keys())

            for i, (param_name, (lower_bound, upper_bound)) in enumerate(
                self.variableParameters.items()
            ):
                ax = axes[i, 0]

                if param_name in self.bestParameters:
                    optimal_value = self.bestParameters[param_name]

                    # Create a single bar for this parameter
                    bar = ax.bar(
                        [0],
                        [upper_bound - lower_bound],
                        bottom=[lower_bound],
                        color="lightblue",
                        alpha=0.7,
                        edgecolor="black",
                        width=0.6,
                    )

                    # Add red dot at optimal position
                    ax.scatter([0], [optimal_value], color="red", s=100, zorder=5)

                    # Add value label on the dot - positioned better to avoid overlap
                    ax.text(
                        0.35,
                        optimal_value,
                        f"{optimal_value:.3f}",
                        ha="left",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )

                    # Set y-axis limits with some padding
                    range_padding = (upper_bound - lower_bound) * 0.05
                    ax.set_ylim(
                        lower_bound - range_padding, upper_bound + range_padding
                    )
                    ax.set_xlim(-0.5, 0.5)

                    # Remove x-axis ticks and labels
                    ax.set_xticks([])

                    # Add parameter name as y-axis label with better formatting
                    ax.set_ylabel(
                        param_name,
                        fontweight="bold",
                        fontsize=11,
                        rotation=0,
                        ha="right",
                        va="center",
                    )

                    # Add grid
                    ax.grid(True, alpha=0.3, axis="y")

                    # Only show title on first subplot
                    if i == 0:
                        ax.set_title(
                            "Optimal Parameter Values within Bounds",
                            fontsize=13,
                            pad=15,
                        )

            # Add legend outside the plot area
            if num_params > 0:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor="lightblue", alpha=0.7, label="Parameter\nRange"),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="red",
                        markersize=10,
                        label="Optimal\nValue",
                    ),
                ]
                # Position legend outside the figure area
                fig.legend(
                    handles=legend_elements,
                    loc="center right",
                    bbox_to_anchor=(0.95, 0.5),
                    fontsize=9,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    handlelength=1.5,
                    handletextpad=0.5,
                    columnspacing=0.5,
                )

            plt.tight_layout(pad=2.0)
            # Adjust layout to make room for legend
            plt.subplots_adjust(right=0.78)
            param_plot_path = os.path.join(
                plots_dir, f"{self.name}-parameter-positions.png"
            )
            plt.savefig(param_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Parameter position plots saved to {param_plot_path}")

        # Try to use new progress manager first, fallback to legacy parsing
        evalNumbersAll = np.array([])
        objectiveValuesAll = np.array([])
        evalNumbersBest = np.array([])
        objectiveValuesBest = np.array([])

        if self.progressManager:
            # Load data using new progress manager
            try:
                self.progressManager.loadData()
                evalNumbersAll, objectiveValuesAll = (
                    self.progressManager.getAllConvergenceData()
                )
                evalNumbersBest, objectiveValuesBest = (
                    self.progressManager.getConvergenceData()
                )
            except Exception as e:
                print(f"Error loading data from progress manager: {e}")
                self.progressManager = None  # Fallback to legacy parsing

        # Fallback to legacy parsing if no progress manager or it failed
        if self.progressManager is None:
            # 2. Convergence plot for progressAll.txt
            progressAllFile = os.path.join(self.runDir, "progressAll.txt")
            if os.path.exists(progressAllFile):
                data = self._parseProgressFile(progressAllFile)
                if data.size > 0:
                    evalNumbersAll = np.arange(1, len(data) + 1)
                    objectiveValuesAll = data[:, -1]

            # 3. Convergence plot for progress.txt
            progressFile = os.path.join(self.runDir, "progress.txt")
            if os.path.exists(progressFile):
                data = self._parseProgressFile(progressFile)
                if data.size > 0:
                    evalNumbersBest = np.arange(1, len(data) + 1)
                    objectiveValuesBest = data[:, -1]

        # Generate convergence plots
        if evalNumbersAll.size > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(evalNumbersAll, objectiveValuesAll, "b-", linewidth=2)
            plt.scatter(
                evalNumbersAll, objectiveValuesAll, color="blue", s=20, alpha=0.6
            )
            plt.xlabel("Evaluation Number")
            plt.ylabel("Objective Function Value")
            plt.title("Convergence History (All Evaluations)")
            plt.grid(True, alpha=0.3)

            convergenceAllPath = os.path.join(
                plots_dir, f"{self.name}-convergence-all.png"
            )
            plt.savefig(convergenceAllPath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Convergence plot (all evaluations) saved to {convergenceAllPath}")

        if evalNumbersBest.size > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(evalNumbersBest, objectiveValuesBest, "g-", linewidth=2)
            plt.scatter(
                evalNumbersBest, objectiveValuesBest, color="green", s=20, alpha=0.6
            )
            plt.xlabel("Evaluation Number")
            plt.ylabel("Best Objective Function Value")
            plt.title("Convergence History (Best Values)")
            plt.grid(True, alpha=0.3)

            convergenceBestPath = os.path.join(
                plots_dir, f"{self.name}-convergence-best.png"
            )
            plt.savefig(convergenceBestPath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Convergence plot (best values) saved to {convergenceBestPath}")

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

    def setStorageFormat(self, storageFormat: str):
        """Set the storage format for progress data (csv or numpy)"""
        if storageFormat.lower() not in ["csv", "numpy"]:
            raise ValueError(f"Unsupported storage format: {storageFormat}")
        self.storageFormat = storageFormat.lower()
        return self

    def migrateLegacyProgressFiles(self):
        """Migrate existing progress.txt and progressAll.txt files to new format"""
        if not self.applied:
            print("Optimization must be applied first")
            return

        # Migrate progress.txt (best evaluations)
        legacyProgressFile = os.path.join(self.runDir, "progress.txt")
        if os.path.exists(legacyProgressFile):
            newProgressFile = os.path.join(self.runDir, "progressBest")

            # Create metadata for migration
            metadata = None
            if hasattr(self, "parameterNames") and self.parameterNames:
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer=self.optimizer,
                    createdTime=datetime.now().isoformat(),
                    description=f"Migrated from legacy progress.txt for {self.name}",
                )

            migrateLegacyProgressFile(
                legacyProgressFile, newProgressFile, self.storageFormat, metadata
            )

        # Migrate progressAll.txt (all evaluations)
        legacyProgressAllFile = os.path.join(self.runDir, "progressAll.txt")
        if os.path.exists(legacyProgressAllFile):
            newProgressAllFile = os.path.join(self.runDir, "progressAll")

            # Create metadata for migration
            metadata = None
            if hasattr(self, "parameterNames") and self.parameterNames:
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer=self.optimizer,
                    createdTime=datetime.now().isoformat(),
                    description=f"Migrated from legacy progressAll.txt for {self.name}",
                )

            migrateLegacyProgressFile(
                legacyProgressAllFile, newProgressAllFile, self.storageFormat, metadata
            )

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

            # Initialize progress manager with metadata
            if hasattr(self, "parameterNames") and self.parameterNames:
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer=self.optimizer,
                    createdTime=datetime.now().isoformat(),
                    description=f"Optimization run for {self.name}",
                )

                progressFilepath = os.path.join(self.runDir, "progressAll")
                self.progressManager = createProgressManager(
                    progressFilepath, self.storageFormat, metadata
                )
                self.progressManager.saveMetadata()
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

                # Save visualization plots if requested
                if self.saveVisualization:
                    self.saveVisualizationPlots()

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise
