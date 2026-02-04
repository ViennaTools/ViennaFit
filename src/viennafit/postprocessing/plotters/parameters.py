"""
Parameter-related plot generation for optimization and GSS results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Optional
from ..base import BasePlotter, StudyData


class ParameterPlotter(BasePlotter):
    """Generates parameter-related plots."""

    def plot(self, data: StudyData, outputDir: str) -> List[str]:
        """Generate parameter plots."""
        createdFiles = []

        # Parameter positions plot (for optimization)
        if data.studyType == "optimization" and data.results:
            filepath = self._plotParameterPositions(data, outputDir)
            if filepath:
                createdFiles.append(filepath)

        # Parameter space exploration plots (for both types)
        if data.evaluationData:
            filepaths = self._plotParameterSpace(data, outputDir)
            createdFiles.extend(filepaths)
        elif data.progressData and data.metadata:
            # Generate parameter space from progress data
            filepaths = self._plotParameterSpaceFromProgress(data, outputDir)
            createdFiles.extend(filepaths)

        return createdFiles

    def _plotParameterPositions(
        self,
        data: StudyData,
        outputDir: str,
        skipParameters: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Plot optimal parameter positions within their bounds using compact horizontal bars."""
        results = data.results
        if not results or "bestParameters" not in results:
            return None

        if skipParameters is None:
            skipParameters = []

        bestParams = results["bestParameters"]

        # Get parameter bounds from metadata or results
        paramBounds = {}
        if data.metadata and "parameterBounds" in data.metadata:
            paramBounds = data.metadata["parameterBounds"]
        elif "variableParameters" in results:
            paramBounds = results["variableParameters"]
        else:
            return None  # No bounds available

        # Filter to only include parameters that have bounds and are not skipped
        boundedParams = {
            name: value
            for name, value in bestParams.items()
            if name in paramBounds and name not in skipParameters
        }

        if not boundedParams:
            return None

        # Use normalized horizontal bar design
        numParams = len(boundedParams)
        figHeight = max(
            3.5, numParams * 0.8 + 1.5
        )  # Slightly more space for annotations
        self._setupPlot(figsize=(10, figHeight))  # Wider for bounds annotations

        # Create normalized horizontal bars (all parameters from 0 to 1)
        paramNames = list(boundedParams.keys())

        for i, (name, value) in enumerate(boundedParams.items()):
            minBound, maxBound = paramBounds[name]
            y_pos = numParams - 1 - i  # Reverse order for top-to-bottom display

            # Calculate normalized position (0 to 1)
            if maxBound != minBound:
                normalizedValue = (value - minBound) / (maxBound - minBound)
            else:
                normalizedValue = 0.5  # Middle if no range

            # Draw full-width normalized range bar (0 to 1)
            plt.barh(
                y_pos,
                1.0,  # Full width = 100% of range
                left=0.0,
                height=0.6,
                color="lightblue",
                alpha=0.7,
                edgecolor="navy",
                linewidth=1,
            )

            # Add optimal value marker at normalized position
            plt.scatter(
                normalizedValue,
                y_pos,
                color="red",
                s=80,
                zorder=5,
                marker="o",
                edgecolor="darkred",
                linewidth=1.5,
            )

            # Add bounds annotations at bar edges
            # Left bound (minimum)
            plt.annotate(
                f"{minBound:.3g}",
                xy=(0.0, y_pos),
                xytext=(-5, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=8,
                color="navy",
                fontweight="bold",
            )

            # Right bound (maximum)
            plt.annotate(
                f"{maxBound:.3g}",
                xy=(1.0, y_pos),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8,
                color="navy",
                fontweight="bold",
            )

            # Add optimal value and percentage annotation above the marker
            percentage = normalizedValue * 100
            plt.annotate(
                f"{value:.3g} ({percentage:.1f}%)",
                xy=(normalizedValue, y_pos),
                xytext=(0, 15),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                ),
            )

        # Format axes with normalized scale
        ax = plt.gca()
        ax.set_xlim(-0.1, 1.1)  # Slight padding for bounds annotations
        ax.set_yticks(range(numParams))
        ax.set_yticklabels(
            [
                self._formatParameterName(paramNames[numParams - 1 - i])
                for i in range(numParams)
            ]
        )

        # Set up normalized x-axis
        plt.xlabel("Normalized Position within Parameter Bounds")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        plt.title("Optimal Parameter Values within Bounds", pad=20)

        # Add grid for easier reading
        plt.grid(True, alpha=0.3, axis="x")
        ax.set_axisbelow(True)

        # Add vertical reference lines at quartiles
        for pos in [0.25, 0.5, 0.75]:
            ax.axvline(x=pos, color="gray", linestyle="--", alpha=0.3, zorder=1)

        # Add legend
        from matplotlib.lines import Line2D

        legendElements = [
            Patch(
                facecolor="lightblue",
                alpha=0.7,
                edgecolor="navy",
                label="Normalized Parameter Range (0-100%)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                markeredgecolor="darkred",
                label="Optimal Value Position",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Reference Lines (25%, 50%, 75%)",
            ),
        ]

        plt.legend(
            handles=legendElements,
            bbox_to_anchor=(1.05, 1),  # Outside plot area, top-right
            loc="upper left",  # Anchor point on legend
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=8,
        )

        # Adjust layout to accommodate external legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for external legend

        filename = f"{data.studyName}-parameter-positions"
        return self._savePlot(outputDir, filename)

    def _plotParameterSpace(self, data: StudyData, outputDir: str) -> List[str]:
        """Plot parameter space exploration from evaluation data."""
        createdFiles = []

        if not data.evaluationData:
            return createdFiles

        # Extract data
        objectives = [result["objective"] for result in data.evaluationData]
        paramNames = list(data.evaluationData[0]["parameters"].keys())

        if len(paramNames) < 1:
            return createdFiles

        # 1. Parameter vs Objective scatter plots
        filepath = self._plotParameterVsObjective(
            data, paramNames, objectives, outputDir
        )
        if filepath:
            createdFiles.append(filepath)

        # 2. 2D parameter space plot (first two parameters)
        if len(paramNames) >= 2:
            filepath = self._plot2dParameterSpace(
                data, paramNames, objectives, outputDir
            )
            if filepath:
                createdFiles.append(filepath)

        return createdFiles

    def _plotParameterVsObjective(
        self,
        data: StudyData,
        paramNames: List[str],
        objectives: List[float],
        outputDir: str,
    ) -> str:
        """Plot each parameter vs objective value."""
        nParams = len(paramNames)
        fig, axes = plt.subplots(
            nParams, 1, figsize=(10, max(6, nParams * 2)), squeeze=False
        )
        plt.subplots_adjust(hspace=0.4)

        for i, paramName in enumerate(paramNames):
            paramValues = [
                result["parameters"][paramName] for result in data.evaluationData
            ]
            ax = axes[i, 0]

            # Color points by objective value
            scatter = ax.scatter(
                paramValues, objectives, c=objectives, cmap="viridis", alpha=0.6, s=20
            )

            formattedName = self._formatParameterName(paramName)
            ax.set_xlabel(formattedName)
            ax.set_ylabel("Objective Value")
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title("Parameter vs Objective Value")
                plt.colorbar(scatter, ax=ax, label="Objective Value")

        filename = f"{data.studyName}-parameter-objective"
        return self._savePlot(outputDir, filename)

    def _plot2dParameterSpace(
        self,
        data: StudyData,
        paramNames: List[str],
        objectives: List[float],
        outputDir: str,
    ) -> str:
        """Plot 2D parameter space using first two parameters."""
        param1Values = [
            result["parameters"][paramNames[0]] for result in data.evaluationData
        ]
        param2Values = [
            result["parameters"][paramNames[1]] for result in data.evaluationData
        ]

        self._setupPlot(figsize=(10, 8))

        scatter = plt.scatter(
            param1Values, param2Values, c=objectives, cmap="viridis", alpha=0.6, s=30
        )
        plt.colorbar(scatter, label="Objective Value")

        formattedName1 = self._formatParameterName(paramNames[0])
        formattedName2 = self._formatParameterName(paramNames[1])
        plt.xlabel(formattedName1)
        plt.ylabel(formattedName2)
        plt.title(f"Parameter Space Exploration: {formattedName1} vs {formattedName2}")
        plt.grid(True, alpha=0.3)

        # Highlight best point if available
        if (
            hasattr(data, "results")
            and data.results
            and "bestParameters" in data.results
        ):
            bestParams = data.results["bestParameters"]
            bestX = bestParams.get(paramNames[0])
            bestY = bestParams.get(paramNames[1])
            bestObj = min(objectives)

            if bestX is not None and bestY is not None:
                plt.scatter(
                    [bestX],
                    [bestY],
                    c="red",
                    s=100,
                    marker="*",
                    edgecolor="black",
                    label=f"Best (Obj: {bestObj:.6f})",
                )
                plt.legend()

        filename = f"{data.studyName}-parameter-space-2d"
        return self._savePlot(outputDir, filename)

    def _plotParameterSpaceFromProgress(
        self, data: StudyData, outputDir: str
    ) -> List[str]:
        """Plot parameter space from progress data when evaluation data not available."""
        # This would be used for optimization runs that don't have detailed evaluation data
        # Implementation would extract parameter values from progress data if available
        # For now, return empty list as this is a lower priority feature
        return []
