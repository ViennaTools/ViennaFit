from .fitProject import Project
from .fitOptimizerWrapper import OptimizerWrapper
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    migrateLegacyProgressFile,
    ProgressDataManager,
)
from .postprocessing import OptimizationPostprocessor
import viennaps as vps
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
import ast
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class Optimization(Study):
    def __init__(self, project: Project):
        super().__init__(project.projectName, project, "optimizationRuns")
        self.optimizer = "dlib"  # Default optimizer
        self.progressManager = None  # Will be initialized in apply()
        self.storageFormat = "csv"  # Default storage format
        self.notes = None  # Optional notes for the optimization run
        # Ax/BoTorch specific configuration
        self.batchSize = 4  # Default batch size for Ax/BoTorch
        self.numBatches = None  # Number of batches for Ax/BoTorch (alternative to numEvaluations)
        self.initialSamples = None  # Will default to 2*numParams if not set

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

        # Update project optimization summary
        try:
            self.project.updateOptimizationSummary()
        except Exception as e:
            print(f"Warning: Could not update optimization summary: {e}")

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
            "notes": self.notes,
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

    def setName(self, name: str):
        """Set the name for the optimization run (only allowed before apply() is called)"""
        if self.applied:
            raise RuntimeError("Cannot change name after optimization has been applied")

        # Generate new directory paths using parent class logic
        newName, newRunDir = self._generateRunDirectory(name, "optimizationRuns")

        # Update name and paths (directories will be created when apply() is called)
        self.name = newName
        self.runDir = newRunDir
        self.progressDir = os.path.join(self.runDir, "progress")

        return self

    def getName(self) -> str:
        """Get the name of the optimization run"""
        return self.name

    def setNotes(self, notes: str):
        """Set notes for the optimization run"""
        self.notes = notes
        return self

    def setBatchSize(self, batchSize: int):
        """
        Set batch size for Ax/BoTorch optimizer (number of parallel candidates per iteration).

        Args:
            batchSize: Number of parameter configurations to evaluate in parallel per iteration.
                      Default is 4. Higher values explore more points per iteration but may
                      require more function evaluations to converge.

        Returns:
            self for method chaining
        """
        if batchSize < 1:
            raise ValueError("Batch size must be at least 1")
        self.batchSize = batchSize
        return self

    def setInitialSamples(self, initialSamples: int):
        """
        Set number of initial Sobol samples for Ax/BoTorch optimizer.

        Args:
            initialSamples: Number of initial quasi-random (Sobol) samples to collect
                           before starting Bayesian optimization with qEI.
                           Default is max(5, 2*numParameters).

        Returns:
            self for method chaining
        """
        if initialSamples < 1:
            raise ValueError("Initial samples must be at least 1")
        self.initialSamples = initialSamples
        return self

    def setNumBatches(self, numBatches: int):
        """
        Set number of BO batches for Ax/BoTorch optimizer (alternative to numEvaluations).

        This provides a clearer way to configure Ax/BoTorch optimization:
        - Total evaluations = initialSamples + (numBatches * batchSize)

        Args:
            numBatches: Number of Bayesian optimization batches to run after initialization.
                       Each batch generates batchSize candidates using qEI.

        Returns:
            self for method chaining

        Example:
            opt.setInitialSamples(10)  # 10 Sobol samples
            opt.setBatchSize(4)        # 4 candidates per batch
            opt.setNumBatches(10)      # 10 BO batches
            # Total evaluations = 10 + (10 * 4) = 50
        """
        if numBatches < 1:
            raise ValueError("Number of batches must be at least 1")
        self.numBatches = numBatches
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
        numEvaluations: int = None,
        saveAllEvaluations: bool = False,
        saveVisualization: bool = True,
        saveAdditionalMetricVisualizations: bool = False,
    ):
        """
        Apply the optimization.

        Args:
            numEvaluations: Number of evaluations to run (required for dlib/nevergrad, ignored for ax/botorch).
                           For Ax/BoTorch, use setNumBatches() instead.
            saveAllEvaluations: Whether to save all evaluations (not just best)
            saveVisualization: Whether to save visualization meshes for the primary metric
            saveAdditionalMetricVisualizations: Whether to save visualization meshes for additional metrics.
                Only applies when saveVisualization=True and for best/all evaluations.
                Default: False (only primary metric visualizations are saved).
        """
        if not self.applied:
            self.validate()

            # Validate numEvaluations based on optimizer
            if self.optimizer in ["ax", "botorch"]:
                # For Ax/BoTorch, numBatches must be set
                if self.numBatches is None:
                    raise ValueError(
                        "For Ax/BoTorch optimizer, you must call setNumBatches() before apply().\n"
                        "Example:\n"
                        "  opt.setNumBatches(10)  # Required for Ax/BoTorch\n"
                        "  opt.apply()            # No numEvaluations needed"
                    )
                # Calculate actual evaluations for saving in config
                initialSamples = self.initialSamples if self.initialSamples else max(5, 2 * len(self.variableParameters))
                self.numEvaluations = initialSamples + (self.numBatches * self.batchSize)
            else:
                # For dlib/nevergrad, numEvaluations must be provided
                if numEvaluations is None:
                    raise ValueError(
                        f"For {self.optimizer} optimizer, you must provide numEvaluations in apply().\n"
                        "Example:\n"
                        "  opt.apply(numEvaluations=100)"
                    )
                self.numEvaluations = numEvaluations

            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations
            self.saveAdditionalMetricVisualizations = saveAdditionalMetricVisualizations

            # Create directories
            os.makedirs(self.runDir, exist_ok=False)
            os.makedirs(self.progressDir, exist_ok=False)

            # Save process sequence to file now that directory exists
            self._saveProcessSequence()

            self.applied = True
            self.evalCounter = 0
            self.saveStartingConfiguration()

            # Save notes to file if provided
            if self.notes is not None:
                notesFile = os.path.join(self.runDir, "notes.txt")
                with open(notesFile, "w") as f:
                    f.write(self.notes)
                print(f"Notes saved to {notesFile}")

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

        # Set optimization start time for total elapsed time tracking
        self.optimizationStartTime = time.time()

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
                    self.generatePlots()

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise

    def generatePlots(
        self, plotTypes: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate plots using the unified postprocessing framework.

        Args:
            plotTypes: List of plot types to generate. Options include:
                       'convergence', 'parameter'. If None, generates all available plots.

        Returns:
            Dictionary mapping plot type names to lists of created file paths.
        """
        if not self.applied:
            print(
                "Warning: Optimization has not been applied yet. Some plots may not be available."
            )

        try:
            postprocessor = OptimizationPostprocessor(self.runDir)
            results = postprocessor.generatePlots(plotTypes)

            totalPlots = sum(len(files) for files in results.values())
            print(f"Generated {totalPlots} plot(s) in {postprocessor.plotsDir}")

            return results

        except Exception as e:
            print(f"Error generating plots: {e}")
            return {}

    def generateSummaryReport(self, outputFile: Optional[str] = None) -> str:
        """
        Generate a summary report of the optimization results.

        Args:
            outputFile: Optional output file path. If None, saves to run directory.

        Returns:
            Path to the generated report file.
        """
        try:
            postprocessor = OptimizationPostprocessor(self.runDir)
            summaryContent = postprocessor.generateSummaryReport()

            if outputFile is None:
                outputFile = os.path.join(self.runDir, f"{self.name}-summary.md")

            with open(outputFile, "w") as f:
                f.write(summaryContent)

            print(f"Summary report generated: {outputFile}")
            return outputFile

        except Exception as e:
            print(f"Error generating summary report: {e}")
            return ""
