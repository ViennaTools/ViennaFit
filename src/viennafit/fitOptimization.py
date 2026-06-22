from .fitProject import Project
from .fitOptimizerWrapper import OptimizerWrapper
from .fitStudy import Study
from .fitUtilities import (
    createProgressManager,
    ProgressMetadata,
    migrateLegacyProgressFile,
    getViennaVersionInfo,
)
from .postprocessing import OptimizationPostprocessor
import os
import json
import shutil
import numpy as np
import ast
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def _ensureHeadlessPlottingBackend():
    """Force a non-interactive matplotlib backend for the duration of a run.

    An optimization run writes plots to disk repeatedly: the parameter-positions
    plot is saved on *every new best* (see fitObjectiveWrapper), plus convergence
    plots at the end. With an interactive backend (TkAgg/QtAgg/...) each savefig
    talks to a live display server. In a headless or SSH-forwarded session that
    connection can drop mid-run, and the next draw triggers a fatal
    "XIO: fatal IO error" that Xlib aborts the *whole process* on -- it cannot be
    caught from Python, so it silently kills a multi-hour optimization (this is
    what aborted runs after only a handful of evaluations). A run never displays
    plots interactively, so Agg is always the correct choice here.

    Set VIENNAFIT_KEEP_BACKEND=1 to opt out (e.g. live plotting in a notebook).
    """
    if os.environ.get("VIENNAFIT_KEEP_BACKEND"):
        return
    try:
        import matplotlib

        nonInteractive = ("agg", "pdf", "ps", "svg", "cairo", "template")
        current = matplotlib.get_backend()
        if current.lower() not in nonInteractive:
            # No figures exist yet at the start of apply(), so force=True is safe.
            matplotlib.use("Agg", force=True)
            print(
                f"[viennafit] matplotlib backend {current!r} -> 'Agg' for this run "
                "(headless-safe plotting; set VIENNAFIT_KEEP_BACKEND=1 to keep it)."
            )
    except Exception as e:
        print(f"[viennafit] could not set headless plotting backend: {e}")


class Optimization(Study):
    def __init__(self, project: Project):
        super().__init__(project.projectName, project, "optimizationRuns")
        self.optimizer = "dlib"  # Default optimizer
        self._progressManager = None  # Will be initialized in apply()
        self.storageFormat = "csv"  # Default storage format
        self.notes = None  # Optional notes for the optimization run
        # Ax/BoTorch specific configuration
        self.batchSize = 4  # Default batch size for Ax/BoTorch
        self.numBatches = (
            None  # Number of batches for Ax/BoTorch (alternative to numEvaluations)
        )
        self.initialSamples = None  # Will default to 2*numParams if not set
        self.initialGuess = None  # Optional warm-start point (set via setInitialGuess())

        # Fold-based cross-validation (set via setFold())
        self._foldName = None
        self._foldDir = None  # folds/{foldName}/ — shared across all runs of this fold
        self._validateDomainNames = []
        self._trainDomainNames = []
        self._foldTrainDomains = None
        self._foldTrainTargets = None
        self._foldValidateDomains = None
        self._foldValidateTargets = None

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
        lowerBounds = []
        upperBounds = []
        for lowerBound, upperBound in self.variableParameters.values():
            lowerBounds.append(lowerBound)
            upperBounds.append(upperBound)
        return lowerBounds, upperBounds

    def _evaluateSubset(
        self,
        paramDict: Dict[str, float],
        initialDomains: Dict,
        targetDomains: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run the process sequence on a domain subset with fixed parameters.

        Used to compute validation scores after optimization. Has no side effects
        on counters, progress files, or best-score tracking.

        Returns:
            (total_score, {domain_name: score})
        """
        from viennaps import Domain
        from .fitDistanceMetrics import DistanceMetric

        domainCopies = {name: Domain(d) for name, d in initialDomains.items()}

        processResult = self.processSequence(domainCopies, paramDict)

        if isinstance(processResult, tuple) and len(processResult) == 2:
            resultDomains, _ = processResult
        else:
            resultDomains = processResult

        if not isinstance(resultDomains, dict):
            raise ValueError(
                "_evaluateSubset requires a multi-domain process sequence "
                "that returns dict[str, Domain]"
            )

        detailedMetric = DistanceMetric.createDetailed(
            self.distanceMetric,
            criticalDimensionRanges=getattr(self, "criticalDimensionRanges", None),
            sparseFieldExpansionWidth=getattr(self, "sparseFieldExpansionWidth", 200),
        )
        return detailedMetric(resultDomains, targetDomains, False, None)

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
            "actualEvaluations": self._evalCounter,
            "earlyStopped": self.earlyStoppedAt is not None,
            "earlyStoppedAtEvaluation": self.earlyStoppedAt,
        }

        # Record fold split in final-results.json when running as a fold
        if self._foldName is not None:
            result["foldName"] = self._foldName
            result["trainDomains"] = self._trainDomainNames
            result["validateDomains"] = self._validateDomainNames

        # Determine validation domains: fold split takes priority over project roles
        if self._foldValidateDomains:
            validationDomains = self._foldValidateDomains
            validationTargets = self._foldValidateTargets
        else:
            validationDomains = self.project.getValidationDomains()
            validationTargets = self.project.getValidationTargets()

        if validationDomains:
            try:
                print("\nEvaluating validation domains with best parameters...")
                validTotal, validPerDomain = self._evaluateSubset(
                    self.bestParameters, validationDomains, validationTargets
                )
                result["validationScore"] = validTotal
                result["validationPerDomainScores"] = validPerDomain
                print(f"  Validation total score: {validTotal:.6f}")
                for name, score in validPerDomain.items():
                    print(f"    {name}: {score:.6f}")
            except Exception as e:
                print(f"Warning: Validation evaluation failed: {e}")

        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)

        # Find and copy all best domain files (handles both single and multi-domain cases)
        projectDomainDir = os.path.join(
            self.project.projectAbsPath, "domains", "optimalDomains"
        )
        os.makedirs(projectDomainDir, exist_ok=True)

        # Pattern matches both single domain and multi-domain files:
        # - Single: {name}-{eval:03d}.vtp
        # - Multi:  {name}-{eval:03d}-{domainName}.vtp
        basePattern = f"{self.name}-{self.bestEvaluationNumber:03d}"
        progressDir = os.path.join(self.runDir, "progress")

        copiedFiles = []
        if os.path.exists(progressDir):
            for filename in os.listdir(progressDir):
                if filename.startswith(basePattern) and filename.endswith(".vtp"):
                    sourcePath = os.path.join(progressDir, filename)
                    targetPath = os.path.join(projectDomainDir, filename)
                    shutil.copy2(sourcePath, targetPath)
                    copiedFiles.append(filename)

        if copiedFiles:
            print(f"Best domain file(s) copied to {projectDomainDir}:")
            for filename in copiedFiles:
                print(f"  - {filename}")
        else:
            print(
                f"Warning: No domain files found matching pattern '{basePattern}*.vtp' in progress directory"
            )

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
        if not self._applied:
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
            "earlyStoppingPatience": getattr(self, "earlyStoppingPatience", None),
            "earlyStoppingMinEvaluations": getattr(
                self, "earlyStoppingMinEvaluations", 0
            ),
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
        if self._applied:
            raise RuntimeError("Cannot change name after optimization has been applied")

        # When inside a fold, runs live in folds/{foldName}/runs/ not optimizationRuns/
        if self._foldName is not None:
            newName, newRunDir = self._generateRunDirectory(
                name, os.path.join("folds", self._foldName, "runs")
            )
        else:
            newName, newRunDir = self._generateRunDirectory(name, "optimizationRuns")

        # Update name and paths (directories will be created when apply() is called)
        self.name = newName
        self.runDir = newRunDir
        self._progressDir = os.path.join(self.runDir, "progress")

        return self

    def getName(self) -> str:
        """Get the name of the optimization run"""
        return self.name

    def setNotes(self, notes: str):
        """Set notes for the optimization run"""
        self.notes = notes
        return self

    def setFold(self, foldName: str, validateDomains: List[str]) -> "Optimization":
        """
        Configure this optimization as a named fold in a cross-validation study.

        Training domains are all project domains *not* listed in validateDomains.
        The run directory is placed under folds/{foldName}/ (not optimizationRuns/)
        and a fold-info.json is written there on apply(), documenting the split.

        Call this before apply(). Do not set project-level domain roles when using
        folds — the split is specified per run here instead.

        Args:
            foldName: Unique name for this fold, e.g. "fold_W1" or "holdout_W3W5".
            validateDomains: Domain names to hold out from optimization. All other
                             domains in the project are used as training domains.

        Returns:
            self for method chaining
        """
        if self._applied:
            raise RuntimeError("Cannot set fold after optimization has been applied")

        allDomains = self.project.initialDomains
        missing = [n for n in validateDomains if n not in allDomains]
        if missing:
            raise ValueError(
                f"Validation domain(s) {missing} not found in project. "
                f"Available: {list(allDomains.keys())}"
            )

        validateSet = set(validateDomains)
        trainDomains = {k: v for k, v in allDomains.items() if k not in validateSet}
        if not trainDomains:
            raise ValueError(
                "No training domains remain after removing validation domains"
            )

        self._foldName = foldName
        self._foldDir = os.path.join(self.project.projectPath, "folds", foldName)
        self._validateDomainNames = list(validateDomains)
        self._trainDomainNames = list(trainDomains.keys())
        self._foldTrainDomains = trainDomains
        self._foldTrainTargets = {
            k: v
            for k, v in self.project.targetLevelSets.items()
            if k not in validateSet
        }
        self._foldValidateDomains = {
            k: v for k, v in allDomains.items() if k in validateSet
        }
        self._foldValidateTargets = {
            k: v for k, v in self.project.targetLevelSets.items() if k in validateSet
        }

        # Redirect the run directory into folds/{foldName}/runs/{runName}/
        # Each fold can have many runs; the fold name stays stable.
        foldRunsType = os.path.join("folds", foldName, "runs")
        currentRunBaseName = self.name
        newName, newRunDir = self._generateRunDirectory(
            currentRunBaseName, foldRunsType
        )
        self.name = newName
        self.runDir = newRunDir
        self._progressDir = os.path.join(self.runDir, "progress")

        return self

    def setInitialGuess(self, guess: Dict[str, float]):
        """
        Warm-start the optimizer at a known parameter point instead of the
        default bounds-midpoint. Currently honoured by the CMA optimizer, which
        centres its initial distribution on this point (clamped to the bounds).

        Args:
            guess: Mapping of variable-parameter name -> starting value. Must
                   cover every variable parameter; extra keys are ignored.

        Returns:
            self for method chaining
        """
        if self._applied:
            raise RuntimeError("Cannot set initial guess after optimization has been applied")
        self.initialGuess = dict(guess)
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

    def setEarlyStopping(
        self, patienceEvaluations: int = None, minEvaluations: int = 0
    ):
        """
        Configure early stopping criterion.

        Args:
            patienceEvaluations: Stop if no improvement for this many evaluations.
                                Set to None to disable.
            minEvaluations: Minimum evaluations before early stopping can trigger.

        Returns:
            self for method chaining
        """
        if patienceEvaluations is not None and patienceEvaluations < 1:
            raise ValueError("patienceEvaluations must be at least 1")
        self.earlyStoppingPatience = patienceEvaluations
        self.earlyStoppingMinEvaluations = minEvaluations
        return self

    def migrateLegacyProgressFiles(self):
        """Migrate existing progress.txt and progressAll.txt files to new format"""
        if not self._applied:
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
        saveComparison: bool = True,
        saveAdditionalMetricVisualizations: bool = False,
    ):
        """
        Apply the optimization.

        Args:
            numEvaluations: Number of evaluations to run (required for dlib/nevergrad, ignored for ax/botorch).
                           For Ax/BoTorch, use setNumBatches() instead.
            saveAllEvaluations: Whether to save all evaluations (not just best)
            saveComparison: Whether to save comparison metric .vtp files for the primary metric.
                Convergence/parameter plots are always generated regardless of this flag.
            saveAdditionalMetricVisualizations: Whether to save visualization meshes for additional metrics.
                Only applies when saveComparison=True and for best/all evaluations.
                Default: False (only primary metric visualizations are saved).
        """
        # A run dumps plots to disk throughout; never let an interactive backend
        # tie the optimization's survival to a live (and droppable) X connection.
        _ensureHeadlessPlottingBackend()

        if not self._applied:
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
                initialSamples = (
                    self.initialSamples
                    if self.initialSamples
                    else max(5, 2 * len(self.variableParameters))
                )
                self.numEvaluations = initialSamples + (
                    self.numBatches * self.batchSize
                )
            else:
                # For dlib/nevergrad, numEvaluations must be provided
                if numEvaluations is None:
                    raise ValueError(
                        f"For {self.optimizer} optimizer, you must provide numEvaluations in apply().\n"
                        "Example:\n"
                        "  opt.apply(numEvaluations=100)"
                    )
                self.numEvaluations = numEvaluations

            self.saveComparison = saveComparison
            self.saveAllEvaluations = saveAllEvaluations
            self.saveAdditionalMetricVisualizations = saveAdditionalMetricVisualizations

            # Create fold directory and write fold-info.json (shared across all runs
            # of this fold; only written once — subsequent runs leave it intact)
            if self._foldName is not None:
                os.makedirs(self._foldDir, exist_ok=True)
                foldInfoPath = os.path.join(self._foldDir, "fold-info.json")
                if not os.path.exists(foldInfoPath):
                    foldInfo = {
                        "foldName": self._foldName,
                        "validateDomains": self._validateDomainNames,
                        "trainDomains": self._trainDomainNames,
                        "createdDate": datetime.now().isoformat(),
                    }
                    with open(foldInfoPath, "w") as f:
                        json.dump(foldInfo, f, indent=4)

            # Create run directories
            os.makedirs(self.runDir, exist_ok=False)
            os.makedirs(self._progressDir, exist_ok=False)

            # Save process sequence to file now that directory exists
            self._saveProcessSequence()

            self._applied = True
            self._evalCounter = 0

            # Save notes to file if provided
            if self.notes is not None:
                notesFile = os.path.join(self.runDir, "notes.txt")
                with open(notesFile, "w") as f:
                    f.write(self.notes)
                print(f"Notes saved to {notesFile}")

            # Initialize progress manager with metadata
            if hasattr(self, "parameterNames") and self.parameterNames:
                versionInfo = getViennaVersionInfo()
                metadata = ProgressMetadata(
                    runName=self.name,
                    parameterNames=self.parameterNames,
                    parameterBounds=self.variableParameters,
                    fixedParameters=self.fixedParameters,
                    optimizer=self.optimizer,
                    createdTime=datetime.now().isoformat(),
                    description=f"Optimization run for {self.name}",
                    numEvaluations=self.numEvaluations,
                    notes=self.notes,
                    viennapsVersion=versionInfo["viennapsVersion"],
                    viennalsVersion=versionInfo["viennalsVersion"],
                    viennapsCommit=versionInfo["viennapsCommit"],
                    viennalsCommit=versionInfo["viennalsCommit"],
                )

                progressFilepath = os.path.join(self.runDir, "progressAll")
                self._progressManager = createProgressManager(
                    progressFilepath, self.storageFormat, metadata
                )
                self._progressManager.saveMetadata()
        else:
            print("Optimization has already been applied.")
            return

        # Set optimization start time for total elapsed time tracking
        self._optimizationStartTime = time.time()

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

                print("Optimization completed successfully:")
                print(f"  Function evaluations: {result['nfev']}")
                print(f"  Best score: {result['fun']:.6f}")
                print("  Best parameters:")
                for name, value in result["x"].items():
                    print(f"    {name}: {value:.6f}")
                print(f" Best evaluation #: {self.bestEvaluationNumber}")

                if result.get("earlyStopped", False):
                    print(
                        f"  Optimization stopped early at evaluation {self.earlyStoppedAt}"
                    )
                    print(
                        f"    (No improvement for {self.earlyStoppingPatience} evaluations)"
                    )

                # Save final results
                self.saveResults(self.name + "-final-results.json")

                # Always generate convergence/parameter plots
                self.generatePlots()

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise

    def _saveBestParameterPositionsPlot(self):
        """Save a parameter-positions plot for the current best, overwriting on each new best."""
        if not self.bestParameters or not self.variableParameters:
            return
        try:
            from .postprocessing.plotters.parameters import ParameterPlotter
            from .postprocessing.base import StudyData, PlotConfig

            plotsDir = os.path.join(self.runDir, "plots")
            os.makedirs(plotsDir, exist_ok=True)

            data = StudyData(
                runDir=self.runDir,
                studyName=self.name,
                studyType="optimization",
                metadata={"parameterBounds": self.variableParameters},
                results={"bestParameters": self.bestParameters},
            )
            ParameterPlotter(PlotConfig())._plotParameterPositions(data, plotsDir)
        except Exception:
            pass

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
        if not self._applied:
            print(
                "Warning: Optimization has not been applied yet. Some plots may not be available."
            )

        try:
            postprocessor = OptimizationPostprocessor(self.runDir)
            results = postprocessor.generatePlots(plotTypes)

            totalPlots = sum(len(files) for files in results.values())
            print(f"Generated {totalPlots} plot(s) in {postprocessor._plotsDir}")

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
