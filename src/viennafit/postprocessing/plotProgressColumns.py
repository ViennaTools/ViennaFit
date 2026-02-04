"""
Utility for plotting arbitrary subsets of columns from ViennaFit progress files.

This module provides a flexible plotting utility for visualizing optimization
progress data from progressAll.csv or progressBest.csv files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Union, Tuple


class ProgressPlotter:
    """
    Flexible plotter for ViennaFit progress data.

    Allows plotting any subset of columns from progressAll.csv or progressBest.csv
    files, with customizable styling and layout options.
    """

    def __init__(self, csvPath: str):
        """
        Initialize the plotter with a progress CSV file.

        Args:
            csvPath: Path to progressAll.csv or progressBest.csv file
        """
        self.csvPath = csvPath
        self._data = pd.read_csv(csvPath)
        self._columns = list(self._data.columns)

    def getAvailableColumns(self) -> List[str]:
        """
        Get list of all available columns in the progress file.

        Returns:
            List of column names
        """
        return self._columns

    def plot(
        self,
        columns: Optional[List[str]] = None,
        xColumn: str = "evaluationNumber",
        logScale: Union[bool, List[str]] = False,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        savePath: Optional[str] = None,
        dpi: int = 300,
        showGrid: bool = True,
        lineWidth: float = 1.5,
        markerSize: float = 15,
        fontSize: float = 1.0,
        shareX: bool = True,
        colors: Optional[List[str]] = None,
    ) -> None:
        """
        Plot selected columns from the progress file.

        Args:
            columns: List of column names to plot. If None, plots all numeric columns
                    except evaluationNumber and elapsedTime
            xColumn: Column to use for x-axis (default: 'evaluationNumber')
            logScale: If True, use log scale for all y-axes. If list of column names,
                     use log scale only for those columns
            figsize: Figure size as (width, height). If None, auto-calculated based on
                    number of subplots
            title: Overall figure title. If None, uses filename
            savePath: Path to save the figure. If None, displays interactively
            dpi: Resolution for saved figure
            showGrid: Whether to show grid on plots
            lineWidth: Width of plot lines
            markerSize: Size of scatter markers
            fontSize: Font size multiplier
            shareX: Whether subplots share x-axis
            colors: List of colors for each column. If None, uses default color cycle

        Returns:
            None (displays or saves the plot)
        """
        # Determine which columns to plot
        if columns is None:
            # Plot all numeric columns except standard metadata
            numericCols = self._data.select_dtypes(include=[np.number]).columns
            excludeCols = ["evaluationNumber", "elapsedTime"]
            columns = [col for col in numericCols if col not in excludeCols]

        # Validate columns
        invalidCols = [col for col in columns if col not in self._columns]
        if invalidCols:
            raise ValueError(
                f"Invalid columns: {invalidCols}. Available: {self._columns}"
            )

        # Validate x column
        if xColumn not in self._columns:
            raise ValueError(f"Invalid x column: {xColumn}. Available: {self._columns}")

        # Setup log scale settings
        if isinstance(logScale, bool):
            logScaleCols = columns if logScale else []
        else:
            logScaleCols = logScale

        # Calculate figure size
        numPlots = len(columns)
        if figsize is None:
            width = 8
            height = max(6, numPlots * 1.5)
            figsize = (width, height)

        # Setup matplotlib style
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        originalFontSize = plt.rcParams["font.size"]
        plt.rcParams["font.size"] = originalFontSize * fontSize

        try:
            # Create figure and subplots
            fig, axs = plt.subplots(
                numPlots, 1, figsize=figsize, dpi=dpi, sharex=shareX
            )

            # Ensure axs is always a list
            if numPlots == 1:
                axs = [axs]

            # Setup default colors if not provided
            if colors is None:
                colors = [
                    "blue",
                    "green",
                    "red",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                ]

            # Get x data
            xData = self._data[xColumn].values

            # Plot each column
            for i, column in enumerate(columns):
                ax = axs[i]
                yData = self._data[column].values
                color = colors[i % len(colors)]

                # Plot line and scatter
                ax.plot(xData, yData, linewidth=lineWidth, color=color)
                ax.scatter(xData, yData, color=color, s=markerSize)

                # Format column name for label
                ylabel = self._formatColumnName(column)
                ax.set_ylabel(ylabel, fontsize=plt.rcParams["font.size"])

                # Apply log scale if requested
                if column in logScaleCols:
                    ax.set_yscale("log")

                # Configure grid
                if showGrid:
                    ax.grid(True, alpha=0.3)

                # Configure tick parameters
                ax.tick_params(axis="both", labelsize=plt.rcParams["font.size"] * 0.8)

                # Set tick locators for cleaner appearance
                if column not in logScaleCols:
                    ax.locator_params(axis="y", nbins=4)

            # Set x-label on bottom subplot
            axs[-1].set_xlabel(
                self._formatColumnName(xColumn), fontsize=plt.rcParams["font.size"]
            )
            axs[-1].locator_params(axis="x", nbins=6)

            # Set title
            if title is None:
                title = os.path.splitext(os.path.basename(self.csvPath))[0]
            fig.suptitle(title, fontsize=plt.rcParams["font.size"] * 1.2)

            # Adjust layout
            plt.tight_layout()

            # Save or show
            if savePath:
                plt.savefig(savePath, dpi=dpi, bbox_inches="tight")
                print(f"Plot saved to: {savePath}")
            else:
                plt.show()

        finally:
            # Restore original font size
            plt.rcParams["font.size"] = originalFontSize
            plt.close(fig)

    def plotMultiAxis(
        self,
        columns: Optional[List[str]] = None,
        xColumn: str = "evaluationNumber",
        individualScale: bool = False,
        scaleMethod: str = "minmax",
        logScale: bool = False,
        logBeforeScale: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        savePath: Optional[str] = None,
        dpi: int = 300,
        showGrid: bool = True,
        lineWidth: float = 1.5,
        markerSize: float = 4,
        fontSize: float = 1.0,
        colors: Optional[List[str]] = None,
        showLegend: bool = True,
        showMarkers: bool = True,
        highlightNonImproving: bool = False,
        highlightColor: str = "red",
        highlightMarker: str = "x",
        highlightSize: float = 8,
        markMinimum: bool = False,
        minimumMarker: str = "*",
        minimumColor: Optional[str] = None,
        minimumSize: float = 15,
        minimumEdgeColor: str = "black",
        minimumEdgeWidth: float = 1.5,
    ) -> None:
        """
        Plot multiple columns on a single axis with shared x-axis.

        Args:
            columns: List of column names to plot. If None, plots all numeric columns
            xColumn: Column to use for x-axis (default: 'evaluationNumber')
            individualScale: If True, scale each column individually to [0, 1] or [-1, 1]
                           If False, plot raw values (useful with logScale=True)
            scaleMethod: Scaling method: 'minmax' (0 to 1), 'standard' (standardize),
                        or 'symmetric' (-1 to 1)
            logScale: If True, use log scale for y-axis (only works when individualScale=False)
            logBeforeScale: If True, apply log10 transformation before scaling
                          (only works when individualScale=True, ignored otherwise)
            figsize: Figure size as (width, height)
            title: Figure title. If None, uses filename
            savePath: Path to save the figure. If None, displays interactively
            dpi: Resolution for saved figure
            showGrid: Whether to show grid on plot
            lineWidth: Width of plot lines
            markerSize: Size of markers on data points
            fontSize: Font size multiplier
            colors: List of colors for each column. If None, uses default color cycle
            showLegend: Whether to show legend
            showMarkers: Whether to show markers on data points
            highlightNonImproving: If True, highlight points that didn't achieve new minimum
            highlightColor: Color for non-improving point markers
            highlightMarker: Marker style for non-improving points ('x', 's', '^', etc.)
            highlightSize: Size of highlight markers
            markMinimum: If True, mark the minimum value of each curve with a special marker
            minimumMarker: Marker style for minimum point ('*', 'D', 'P', etc.)
            minimumColor: Color for minimum marker. If None, uses the same color as the curve
            minimumSize: Size of minimum marker
            minimumEdgeColor: Edge color for minimum marker
            minimumEdgeWidth: Edge width for minimum marker

        Returns:
            None (displays or saves the plot)
        """
        # Determine which columns to plot
        if columns is None:
            numericCols = self._data.select_dtypes(include=[np.number]).columns
            excludeCols = ["evaluationNumber", "elapsedTime"]
            columns = [col for col in numericCols if col not in excludeCols]

        # Validate columns
        invalidCols = [col for col in columns if col not in self._columns]
        if invalidCols:
            raise ValueError(
                f"Invalid columns: {invalidCols}. Available: {self._columns}"
            )

        # Validate conflicting options
        if logScale and individualScale:
            import warnings

            warnings.warn(
                "Using logScale=True with individualScale=True may produce confusing results. "
                "Consider using logBeforeScale=True instead to apply log transformation before normalization.",
                UserWarning,
            )

        # Setup matplotlib style
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        originalFontSize = plt.rcParams["font.size"]
        plt.rcParams["font.size"] = originalFontSize * fontSize

        # Setup default colors if not provided
        if colors is None:
            colors = plt.cm.tab10.colors

        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Get x data
            xData = self._data[xColumn].values

            # Plot each column
            for i, column in enumerate(columns):
                yDataOriginal = self._data[column].values.copy()
                yData = yDataOriginal.copy()
                color = colors[i % len(colors)]
                label = self._formatColumnName(column)

                # Identify non-improving points (before any transformation)
                nonImprovingMask = None
                if highlightNonImproving:
                    nonImprovingMask = self._findNonImprovingPoints(yDataOriginal)

                # Apply log transformation before scaling if requested
                if logBeforeScale and individualScale:
                    yData = self._applyLogTransform(yData)

                # Apply individual scaling if requested
                if individualScale:
                    yData = self._scaleData(yData, scaleMethod)

                # Plot with or without markers
                if showMarkers:
                    ax.plot(
                        xData,
                        yData,
                        linewidth=lineWidth,
                        color=color,
                        label=label,
                        marker="o",
                        markersize=markerSize,
                    )
                else:
                    ax.plot(xData, yData, linewidth=lineWidth, color=color, label=label)

                # Highlight non-improving points if requested
                if highlightNonImproving and nonImprovingMask is not None:
                    # Transform the non-improving y-values the same way as main data
                    yDataHighlight = yDataOriginal.copy()
                    if logBeforeScale and individualScale:
                        yDataHighlight = self._applyLogTransform(yDataHighlight)
                    if individualScale:
                        yDataHighlight = self._scaleData(yDataHighlight, scaleMethod)

                    # Plot highlighted markers
                    ax.scatter(
                        xData[nonImprovingMask],
                        yDataHighlight[nonImprovingMask],
                        marker=highlightMarker,
                        s=highlightSize**2,
                        color=highlightColor,
                        zorder=5,
                        alpha=0.7,
                    )

                # Mark minimum value if requested
                if markMinimum:
                    # Find minimum in original data
                    minIdx = np.argmin(yDataOriginal)
                    minX = xData[minIdx]

                    # Transform the minimum y-value the same way as main data
                    yDataMin = yDataOriginal.copy()
                    if logBeforeScale and individualScale:
                        yDataMin = self._applyLogTransform(yDataMin)
                    if individualScale:
                        yDataMin = self._scaleData(yDataMin, scaleMethod)

                    minY = yDataMin[minIdx]

                    # Use curve color if minimumColor not specified
                    markerColor = minimumColor if minimumColor is not None else color

                    # Plot minimum marker
                    ax.scatter(
                        minX,
                        minY,
                        marker=minimumMarker,
                        s=minimumSize**2,
                        color=markerColor,
                        edgecolors=minimumEdgeColor,
                        linewidths=minimumEdgeWidth,
                        zorder=10,
                        label=f"Min: {label}",
                    )

            # Configure axes
            ax.set_xlabel(
                self._formatColumnName(xColumn), fontsize=plt.rcParams["font.size"]
            )

            if individualScale:
                logPrefix = "Log-Scaled + " if logBeforeScale else ""
                if scaleMethod == "minmax":
                    ax.set_ylabel(
                        f"{logPrefix}Normalized Value [0, 1]",
                        fontsize=plt.rcParams["font.size"],
                    )
                elif scaleMethod == "symmetric":
                    ax.set_ylabel(
                        f"{logPrefix}Normalized Value [-1, 1]",
                        fontsize=plt.rcParams["font.size"],
                    )
                else:
                    ax.set_ylabel(
                        f"{logPrefix}Standardized Value",
                        fontsize=plt.rcParams["font.size"],
                    )
            else:
                ax.set_ylabel("Value", fontsize=plt.rcParams["font.size"])

            # Apply log scale if requested (note: this is applied AFTER scaling)
            if logScale and not individualScale:
                ax.set_yscale("log")

            # Configure grid and legend
            if showGrid:
                ax.grid(True, alpha=0.3)
            if showLegend:
                ax.legend(fontsize=plt.rcParams["font.size"] * 0.8, loc="best")

            # Configure ticks
            ax.tick_params(axis="both", labelsize=plt.rcParams["font.size"] * 0.8)

            # Set title
            if title is None:
                title = os.path.splitext(os.path.basename(self.csvPath))[0]
            ax.set_title(title, fontsize=plt.rcParams["font.size"] * 1.2)

            # Adjust layout
            plt.tight_layout()

            # Save or show
            if savePath:
                plt.savefig(savePath, dpi=dpi, bbox_inches="tight")
                print(f"Plot saved to: {savePath}")
            else:
                plt.show()

        finally:
            # Restore original font size
            plt.rcParams["font.size"] = originalFontSize
            plt.close(fig)

    def _applyLogTransform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply log10 transformation to data, handling edge cases.

        Args:
            data: Input data array

        Returns:
            Log-transformed data array
        """
        # Handle negative and zero values by shifting data to positive range
        minVal = np.min(data)
        if minVal <= 0:
            # Shift data to be positive, adding a small offset
            offset = abs(minVal) + 1e-10
            data = data + offset

        # Apply log10 transformation
        return np.log10(data)

    def _scaleData(self, data: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Scale data using specified method.

        Args:
            data: Input data array
            method: Scaling method ('minmax', 'standard', or 'symmetric')

        Returns:
            Scaled data array
        """
        if method == "minmax":
            # Scale to [0, 1]
            minVal = np.min(data)
            maxVal = np.max(data)
            if maxVal - minVal == 0:
                return np.zeros_like(data)
            return (data - minVal) / (maxVal - minVal)

        elif method == "symmetric":
            # Scale to [-1, 1]
            minVal = np.min(data)
            maxVal = np.max(data)
            if maxVal - minVal == 0:
                return np.zeros_like(data)
            return 2 * (data - minVal) / (maxVal - minVal) - 1

        elif method == "standard":
            # Standardize (mean=0, std=1)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std

        else:
            raise ValueError(
                f"Unknown scaling method: {method}. Use 'minmax', 'symmetric', or 'standard'"
            )

    def _findNonImprovingPoints(self, data: np.ndarray) -> np.ndarray:
        """
        Find points that are higher than the previously achieved minimum.

        Args:
            data: Input data array (assumes lower values are better)

        Returns:
            Boolean mask where True indicates non-improving points
        """
        nonImprovingMask = np.zeros(len(data), dtype=bool)
        currentMin = float("inf")

        for i, value in enumerate(data):
            if value < currentMin:
                # This is an improving point (new minimum)
                currentMin = value
                nonImprovingMask[i] = False
            else:
                # This is a non-improving point
                nonImprovingMask[i] = True

        return nonImprovingMask

    def _formatColumnName(self, columnName: str) -> str:
        """
        Format column name for display.

        Args:
            columnName: Raw column name

        Returns:
            Formatted column name
        """
        # Handle special cases
        replacements = {
            "objectiveValue": "Objective Value",
            "evaluationNumber": "Evaluation Number",
            "elapsedTime": "Elapsed Time (s)",
        }

        if columnName in replacements:
            return replacements[columnName]

        # Insert spaces before capital letters and numbers
        formatted = ""
        for i, char in enumerate(columnName):
            if i > 0 and (
                char.isupper() or (char.isdigit() and not columnName[i - 1].isdigit())
            ):
                formatted += " "
            formatted += char

        # Capitalize first letter
        formatted = formatted[0].upper() + formatted[1:] if formatted else formatted

        return formatted


def plotProgressFile(
    csvPath: str,
    columns: Optional[List[str]] = None,
    xColumn: str = "evaluationNumber",
    multiAxis: bool = False,
    **kwargs,
) -> None:
    """
    Convenience function to quickly plot a progress file.

    Args:
        csvPath: Path to progressAll.csv or progressBest.csv
        columns: List of columns to plot (None = all numeric columns)
        xColumn: Column to use for x-axis
        multiAxis: If True, plot all columns on single axis. If False, use subplots
        **kwargs: Additional arguments passed to plot() or plotMultiAxis()

    Returns:
        None
    """
    plotter = ProgressPlotter(csvPath)

    if multiAxis:
        plotter.plotMultiAxis(columns=columns, xColumn=xColumn, **kwargs)
    else:
        plotter.plot(columns=columns, xColumn=xColumn, **kwargs)


def printAvailableColumns(csvPath: str) -> None:
    """
    Print all available columns in a progress file.

    Args:
        csvPath: Path to progressAll.csv or progressBest.csv

    Returns:
        None (prints to console)
    """
    plotter = ProgressPlotter(csvPath)
    columns = plotter.getAvailableColumns()

    print(f"\nAvailable columns in {os.path.basename(csvPath)}:")
    print("-" * 60)
    for i, col in enumerate(columns, 1):
        print(f"{i:3d}. {col}")
    print("-" * 60)
    print(f"Total: {len(columns)} columns\n")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python plotProgressColumns.py <path_to_csv> [column1] [column2] ..."
        )
        print("\nExamples:")
        print("  python plotProgressColumns.py progressAll.csv")
        print("  python plotProgressColumns.py progressBest.csv objectiveValue")
        print(
            "  python plotProgressColumns.py progressAll.csv param1 param2 objectiveValue"
        )
        sys.exit(1)

    csvPath = sys.argv[1]

    if not os.path.exists(csvPath):
        print(f"Error: File not found: {csvPath}")
        sys.exit(1)

    # If no columns specified, show available columns and plot all
    if len(sys.argv) == 2:
        printAvailableColumns(csvPath)
        print("Plotting all numeric columns...")
        plotProgressFile(csvPath)
    else:
        # Plot specific columns
        columns = sys.argv[2:]
        print(f"Plotting columns: {', '.join(columns)}")
        plotProgressFile(csvPath, columns=columns)
