"""
Optimization runs summary and aggregation functionality.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import glob


class RunsSummary:
    """Aggregates and manages optimization run results for a project."""

    def __init__(self, projectPath: str):
        """
        Initialize runs summary for a project.

        Args:
            projectPath: Absolute path to the project directory
        """
        self._projectPath = os.path.abspath(projectPath)
        self._projectName = os.path.basename(self._projectPath)
        self._optimizationRunsDir = os.path.join(self._projectPath, "optimizationRuns")
        self._summaryFilePath = os.path.join(
            self._projectPath, f"{self._projectName}-optimization-summary.json"
        )
        self._runs = []
        self._bestRun = None

    def scanRuns(self) -> int:
        """
        Scan optimization runs directory and load all final results.

        Returns:
            Number of runs found
        """
        self._runs = []

        if not os.path.exists(self._optimizationRunsDir):
            print(f"No optimizationRuns directory found in {self._projectPath}")
            return 0

        # Find all final-results.json files
        resultsPattern = os.path.join(
            self._optimizationRunsDir, "**", "*-final-results.json"
        )
        resultsFiles = glob.glob(resultsPattern, recursive=True)

        for resultsFile in resultsFiles:
            try:
                with open(resultsFile, "r") as f:
                    results = json.load(f)

                runDir = os.path.dirname(resultsFile)
                runName = os.path.basename(runDir)

                # Get run metadata
                startConfigFile = os.path.join(
                    runDir, f"{runName}-startingConfiguration.json"
                )
                notes = None
                createdTime = None

                if os.path.exists(startConfigFile):
                    try:
                        with open(startConfigFile, "r") as f:
                            config = json.load(f)
                            notes = config.get("notes", None)
                    except Exception:
                        pass

                # Get directory creation time as fallback
                try:
                    createdTime = datetime.fromtimestamp(
                        os.path.getctime(runDir)
                    ).isoformat()
                except Exception:
                    createdTime = None

                # Aggregate run info
                runInfo = {
                    "runName": runName,
                    "runDir": runDir,
                    "resultsFile": resultsFile,
                    "bestScore": results.get("bestScore", None),
                    "bestEvaluationNumber": results.get("bestEvaluation#", None),
                    "bestParameters": results.get("bestParameters", {}),
                    "fixedParameters": results.get("fixedParameters", {}),
                    "variableParameters": results.get("variableParameters", {}),
                    "optimizer": results.get("optimizer", "unknown"),
                    "numEvaluations": results.get("numEvaluations", None),
                    "notes": notes,
                    "createdTime": createdTime,
                }

                self._runs.append(runInfo)

            except Exception as e:
                print(f"Warning: Could not load results from {resultsFile}: {e}")

        # Sort runs by creation time (most recent first)
        self._runs.sort(key=lambda x: x.get("createdTime", ""), reverse=True)

        # Identify best run (lowest bestScore)
        if self._runs:
            validRuns = [r for r in self._runs if r["bestScore"] is not None]
            if validRuns:
                self._bestRun = min(validRuns, key=lambda x: x["bestScore"])

        print(f"Found {len(self._runs)} optimization run(s)")
        return len(self._runs)

    def getBestRun(self) -> Optional[Dict]:
        """
        Get information about the best optimization run.

        Returns:
            Dictionary with best run information, or None if no runs found
        """
        return self._bestRun

    def getRuns(self) -> List[Dict]:
        """
        Get list of all optimization runs.

        Returns:
            List of dictionaries with run information
        """
        return self._runs

    def getRunByName(self, runName: str) -> Optional[Dict]:
        """
        Get information about a specific run by name.

        Args:
            runName: Name of the optimization run

        Returns:
            Dictionary with run information, or None if not found
        """
        for run in self._runs:
            if run["runName"] == runName:
                return run
        return None

    def saveSummary(self):
        """Save the runs summary to a JSON file."""
        if not self._runs:
            print("No runs to save")
            return

        summary = {
            "projectName": self._projectName,
            "projectPath": self._projectPath,
            "totalRuns": len(self._runs),
            "bestRun": (
                {
                    "runName": self._bestRun["runName"],
                    "bestScore": self._bestRun["bestScore"],
                    "bestEvaluationNumber": self._bestRun["bestEvaluationNumber"],
                    "createdTime": self._bestRun.get("createdTime", None),
                }
                if self._bestRun
                else None
            ),
            "lastUpdated": datetime.now().isoformat(),
            "runs": self._runs,
        }

        with open(self._summaryFilePath, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Summary saved to {self._summaryFilePath}")

    def loadSummary(self) -> bool:
        """
        Load existing summary from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self._summaryFilePath):
            return False

        try:
            with open(self._summaryFilePath, "r") as f:
                summary = json.load(f)

            self._runs = summary.get("runs", [])
            bestRunName = (
                summary.get("bestRun", {}).get("runName", None)
                if summary.get("bestRun")
                else None
            )

            if bestRunName:
                self._bestRun = self.getRunByName(bestRunName)

            print(f"Loaded summary with {len(self._runs)} run(s)")
            return True

        except Exception as e:
            print(f"Error loading summary: {e}")
            return False

    def updateSummary(self):
        """Re-scan runs and update the summary file."""
        self.scanRuns()
        self.saveSummary()

    def generateMarkdownReport(self, outputPath: Optional[str] = None) -> str:
        """
        Generate a markdown report of optimization runs.

        Args:
            outputPath: Optional path to save the report. If None, returns the report as string.

        Returns:
            Path to the saved report file, or the report content as string
        """
        if not self._runs:
            return "No optimization runs found."

        # Generate markdown content
        lines = [
            f"# Optimization Runs Overview: {self._projectName}",
            "",
            f"**Total runs:** {len(self._runs)}",
            f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if self._bestRun:
            lines.extend(
                [
                    "## Best Run",
                    "",
                    f"- **Run name:** {self._bestRun['runName']}",
                    f"- **Best score:** {self._bestRun['bestScore']:.6f}",
                    f"- **Best evaluation #:** {self._bestRun['bestEvaluationNumber']}",
                    f"- **Created:** {self._bestRun.get('createdTime', 'Unknown')}",
                    "",
                ]
            )

            if self._bestRun.get("notes"):
                lines.extend(
                    [
                        f"**Notes:** {self._bestRun['notes']}",
                        "",
                    ]
                )

            lines.extend(["### Best Parameters", ""])
            for param, value in self._bestRun["bestParameters"].items():
                lines.append(f"- **{param}:** {value}")
            lines.append("")

        # All runs table
        lines.extend(
            [
                "## All Optimization Runs",
                "",
                "| Run Name | Best Score | Evaluations | Best Eval # | Optimizer | Created |",
                "|----------|------------|-------------|-------------|-----------|---------|",
            ]
        )

        # Sort by best score for the table
        sortedRuns = sorted(
            [r for r in self._runs if r["bestScore"] is not None],
            key=lambda x: x["bestScore"],
        )

        for run in sortedRuns:
            runName = run["runName"]
            bestScore = (
                f"{run['bestScore']:.6f}" if run["bestScore"] is not None else "N/A"
            )
            numEvals = run.get("numEvaluations", "N/A")
            bestEvalNum = run.get("bestEvaluationNumber", "N/A")
            optimizer = run.get("optimizer", "unknown")
            created = run.get("createdTime", "Unknown")
            if created != "Unknown":
                try:
                    created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            # Mark best run with ⭐
            marker = (
                " ⭐"
                if self._bestRun and run["runName"] == self._bestRun["runName"]
                else ""
            )
            lines.append(
                f"| {runName}{marker} | {bestScore} | {numEvals} | {bestEvalNum} | {optimizer} | {created} |"
            )

        report = "\n".join(lines)

        if outputPath:
            with open(outputPath, "w") as f:
                f.write(report)
            print(f"Markdown report saved to {outputPath}")
            return outputPath
        else:
            return report

    def printSummary(self):
        """Print a formatted summary of all optimization runs to console."""
        if not self._runs:
            print("No optimization runs found.")
            return

        print("\n" + "=" * 80)
        print(f"Optimization Runs Overview: {self._projectName}")
        print("=" * 80)
        print(f"Total runs: {len(self._runs)}")

        if self._bestRun:
            print("\n" + "-" * 80)
            print("BEST RUN")
            print("-" * 80)
            print(f"  Run name:          {self._bestRun['runName']}")
            print(f"  Best score:        {self._bestRun['bestScore']:.6f}")
            print(f"  Best evaluation #: {self._bestRun['bestEvaluationNumber']}")
            print(f"  Created:           {self._bestRun.get('createdTime', 'Unknown')}")

            if self._bestRun.get("notes"):
                print(f"  Notes:             {self._bestRun['notes']}")

        print("\n" + "-" * 80)
        print("ALL RUNS (sorted by best score)")
        print("-" * 80)

        # Sort by best score
        sortedRuns = sorted(
            [r for r in self._runs if r["bestScore"] is not None],
            key=lambda x: x["bestScore"],
        )

        # Print table header
        print(f"{'Rank':<6} {'Run Name':<40} {'Best Score':<15} {'Best Eval #':<12}")
        print("-" * 80)

        for i, run in enumerate(sortedRuns, 1):
            marker = (
                "⭐"
                if self._bestRun and run["runName"] == self._bestRun["runName"]
                else ""
            )
            runName = run["runName"][:38]  # Truncate if too long
            bestScore = f"{run['bestScore']:.6f}"
            bestEvalNum = run.get("bestEvaluationNumber", "N/A")

            print(f"{i:<6} {runName:<40} {bestScore:<15} {bestEvalNum:<12} {marker}")

        print("=" * 80 + "\n")

    def compareRuns(self, runNames: List[str]) -> Dict:
        """
        Compare specific runs and return comparison data.

        Args:
            runNames: List of run names to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {"runs": [], "parameters": {}}

        for runName in runNames:
            run = self.getRunByName(runName)
            if run:
                comparison["runs"].append(
                    {
                        "runName": run["runName"],
                        "bestScore": run["bestScore"],
                        "bestEvaluationNumber": run["bestEvaluationNumber"],
                        "bestParameters": run["bestParameters"],
                    }
                )

                # Collect all parameter names
                for param in run["bestParameters"].keys():
                    if param not in comparison["parameters"]:
                        comparison["parameters"][param] = {}
                    comparison["parameters"][param][runName] = run["bestParameters"][
                        param
                    ]

        return comparison
