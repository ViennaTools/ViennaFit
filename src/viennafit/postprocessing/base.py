"""
Base classes and interfaces for the postprocessing framework.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfig:
    """Configuration for plot generation."""

    dpi: int = 300
    figsize: Tuple[int, int] = (10, 6)
    style: str = "default"
    saveFormat: str = "png"
    bboxInches: str = "tight"


@dataclass
class StudyData:
    """Container for study data and metadata."""

    runDir: str
    studyName: str
    studyType: str  # 'optimization' or 'global_sensitivity'
    metadata: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    progressData: Optional[Dict[str, Any]] = None
    evaluationData: Optional[List[Dict[str, Any]]] = None
    sensitivityData: Optional[Dict[str, Any]] = None


class BasePlotter(ABC):
    """Abstract base class for all plot generators."""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()

    @abstractmethod
    def plot(self, data: StudyData, outputDir: str) -> List[str]:
        """
        Generate plots for the given study data.

        Args:
            data: Study data container
            outputDir: Directory to save plots

        Returns:
            List of created plot file paths
        """
        pass

    def _setupPlot(self, figsize: Optional[Tuple[int, int]] = None):
        """Setup matplotlib plot with common styling."""
        plt.style.use(self.config.style)
        figSize = figsize or self.config.figsize
        plt.figure(figsize=figSize)

    def _savePlot(self, outputDir: str, filename: str) -> str:
        """Save current plot to file."""
        filepath = os.path.join(outputDir, f"{filename}.{self.config.saveFormat}")
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches=self.config.bboxInches)
        plt.close()
        return filepath

    def _formatParameterName(self, paramName: str) -> str:
        """Format parameter name for display."""
        # Replace underscores with spaces and capitalize
        return paramName.replace("_", " ").title()


class BasePostprocessor(ABC):
    """Abstract base class for study postprocessors."""

    def __init__(self, runDir: str, config: Optional[PlotConfig] = None):
        self.runDir = os.path.abspath(runDir)
        self.config = config or PlotConfig()
        self.studyName = os.path.basename(self.runDir)
        self._plotsDir = os.path.join(self.runDir, "plots")

        # Ensure plots directory exists
        os.makedirs(self._plotsDir, exist_ok=True)

        # Available plotters
        self._plotters: List[BasePlotter] = []

    @abstractmethod
    def loadData(self) -> StudyData:
        """Load all available data for this study."""
        pass

    @abstractmethod
    def getAvailablePlotters(self) -> List[BasePlotter]:
        """Get list of applicable plotters for this study type."""
        pass

    def generatePlots(
        self, plotTypes: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate plots for this study.

        Args:
            plotTypes: List of plot types to generate. If None, generates all available plots.

        Returns:
            Dictionary mapping plot type names to lists of created file paths.
        """
        # Load study data
        data = self.loadData()

        # Get available plotters
        availablePlotters = self.getAvailablePlotters()

        # Filter plotters if specific types requested
        if plotTypes:
            plottersToUse = [
                p
                for p in availablePlotters
                if p.__class__.__name__.replace("Plotter", "").lower() in plotTypes
            ]
        else:
            plottersToUse = availablePlotters

        # Generate plots
        results = {}
        for plotter in plottersToUse:
            plotterName = plotter.__class__.__name__.replace("Plotter", "").lower()
            try:
                createdFiles = plotter.plot(data, self._plotsDir)
                results[plotterName] = createdFiles
                print(f"Generated {len(createdFiles)} {plotterName} plot(s)")
            except Exception as e:
                print(f"Warning: Failed to generate {plotterName} plots: {e}")
                results[plotterName] = []

        return results

    def getSummary(self) -> Dict[str, Any]:
        """Get summary information about this study."""
        data = self.loadData()

        summary = {
            "studyName": data.studyName,
            "studyType": data.studyType,
            "runDirectory": data.runDir,
        }

        # Add metadata if available
        if data.metadata:
            summary.update(data.metadata)

        return summary
