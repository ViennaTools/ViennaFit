"""
Utility functions for postprocessing operations.
"""

import os
import glob
from typing import List, Dict, Optional, Tuple
from .base import PlotConfig
from .managers import OptimizationPostprocessor, GSSPostprocessor


def createPostprocessor(runDir: str, config: Optional[PlotConfig] = None):
    """
    Create appropriate postprocessor for a given run directory.

    Args:
        runDir: Path to run directory
        config: Optional plot configuration

    Returns:
        Postprocessor instance (OptimizationPostprocessor or GSSPostprocessor)
    """
    from .loaders import ResultsLoader

    loader = ResultsLoader(runDir)
    studyType = loader._detectStudyType()

    if studyType == "optimization":
        return OptimizationPostprocessor(runDir, config)
    else:
        return GSSPostprocessor(runDir, config)


def findStudyDirectories(projectDir: str, studyType: Optional[str] = None) -> List[str]:
    """
    Find all study directories in a project directory.

    Args:
        projectDir: Path to project directory
        studyType: Optional filter for study type ('optimization' or 'global_sensitivity')

    Returns:
        List of study directory paths
    """
    studyDirs = []

    # Common patterns for study directories
    patterns = [
        os.path.join(projectDir, "optimizationRuns", "*"),
        os.path.join(projectDir, "globalSensStudies", "*"),
        os.path.join(projectDir, "locSensStudies", "*"),
    ]

    for pattern in patterns:
        dirs = glob.glob(pattern)
        for dirPath in dirs:
            if os.path.isdir(dirPath):
                # Filter by study type if specified
                if studyType:
                    from .loaders import ResultsLoader

                    try:
                        loader = ResultsLoader(dirPath)
                        detectedType = loader._detectStudyType()
                        if (
                            studyType == "optimization"
                            and detectedType == "optimization"
                        ) or (
                            studyType == "global_sensitivity"
                            and detectedType == "global_sensitivity"
                        ):
                            studyDirs.append(dirPath)
                    except:
                        continue
                else:
                    studyDirs.append(dirPath)

    return sorted(studyDirs)


def batchGeneratePlots(
    runDirs: List[str],
    plotTypes: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate plots for multiple runs in batch.

    Args:
        runDirs: List of run directory paths
        plotTypes: Optional list of plot types to generate
        config: Optional plot configuration

    Returns:
        Dictionary mapping run names to plot results
    """
    results = {}

    for runDir in runDirs:
        runName = os.path.basename(runDir)
        try:
            postprocessor = createPostprocessor(runDir, config)
            plotResults = postprocessor.generatePlots(plotTypes)
            results[runName] = plotResults
            print(f"Generated plots for {runName}")
        except Exception as e:
            print(f"Error generating plots for {runName}: {e}")
            results[runName] = {}

    return results


def validateRunDirectory(runDir: str) -> Tuple[bool, List[str]]:
    """
    Validate a run directory and return information about available data.

    Args:
        runDir: Path to run directory

    Returns:
        Tuple of (isValid, listOfAvailableDataTypes)
    """
    if not os.path.isdir(runDir):
        return False, ["Directory does not exist"]

    from .loaders import ResultsLoader

    try:
        loader = ResultsLoader(runDir)
        data = loader.loadStudyData()

        availableData = []

        if data.metadata:
            availableData.append("metadata")
        if data.results:
            availableData.append("results")
        if data.progressData:
            availableData.append("progressData")
        if data.evaluationData:
            availableData.append("evaluationData")
        if data.sensitivityData:
            availableData.append("sensitivityData")

        isValid = len(availableData) > 0
        return isValid, availableData

    except Exception as e:
        return False, [f"Error loading data: {str(e)}"]


def getStudySummary(runDir: str) -> Dict[str, any]:
    """
    Get a quick summary of a study without generating plots.

    Args:
        runDir: Path to run directory

    Returns:
        Dictionary containing study summary information
    """
    try:
        postprocessor = createPostprocessor(runDir)
        return postprocessor.getSummary()
    except Exception as e:
        return {
            "error": str(e),
            "runDirectory": runDir,
            "studyName": os.path.basename(runDir),
        }


def cleanupOldPlots(runDir: str, keepLatest: int = 1) -> int:
    """
    Clean up old plot files, keeping only the most recent ones.

    Args:
        runDir: Path to run directory
        keepLatest: Number of latest plot sets to keep

    Returns:
        Number of files deleted
    """
    plotsDir = os.path.join(runDir, "plots")
    if not os.path.exists(plotsDir):
        return 0

    # Get all PNG files with their modification times
    pngFiles = []
    for file in os.listdir(plotsDir):
        if file.endswith(".png"):
            filepath = os.path.join(plotsDir, file)
            mtime = os.path.getmtime(filepath)
            pngFiles.append((filepath, mtime))

    # Sort by modification time (newest first)
    pngFiles.sort(key=lambda x: x[1], reverse=True)

    # Delete files beyond the keepLatest count
    deletedCount = 0
    if len(pngFiles) > keepLatest:
        for filepath, _ in pngFiles[keepLatest:]:
            try:
                os.remove(filepath)
                deletedCount += 1
            except Exception as e:
                print(f"Warning: Could not delete {filepath}: {e}")

    return deletedCount


def exportSummaryReport(runDir: str, outputFile: Optional[str] = None) -> str:
    """
    Export a summary report for a study.

    Args:
        runDir: Path to run directory
        outputFile: Optional output file path. If None, saves to run directory.

    Returns:
        Path to the generated report file
    """
    postprocessor = createPostprocessor(runDir)

    if hasattr(postprocessor, "generateSummaryReport"):
        content = postprocessor.generateSummaryReport()
    elif hasattr(postprocessor, "generateSensitivitySummary"):
        content = postprocessor.generateSensitivitySummary()
    else:
        # Fallback generic summary
        summary = postprocessor.getSummary()
        content = f"# Study Summary: {summary['studyName']}\n\n"
        for key, value in summary.items():
            content += f"**{key}**: {value}\n"

    if outputFile is None:
        outputFile = os.path.join(runDir, f"{postprocessor.studyName}-summary.md")

    with open(outputFile, "w") as f:
        f.write(content)

    return outputFile
