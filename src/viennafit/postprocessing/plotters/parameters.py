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
        if data.studyType == 'optimization' and data.results:
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
        
    def _plotParameterPositions(self, data: StudyData, outputDir: str) -> Optional[str]:
        """Plot optimal parameter positions within their bounds."""
        results = data.results
        if not results or 'bestParameters' not in results:
            return None
            
        bestParams = results['bestParameters']
        
        # Get parameter bounds from metadata or results
        paramBounds = {}
        if data.metadata and 'parameterBounds' in data.metadata:
            paramBounds = data.metadata['parameterBounds']
        elif 'variableParameters' in results:
            paramBounds = results['variableParameters']
        else:
            return None  # No bounds available
            
        # Filter to only include parameters that have bounds
        boundedParams = {
            name: value for name, value in bestParams.items()
            if name in paramBounds
        }
        
        if not boundedParams:
            return None
            
        numParams = len(boundedParams)
        fig, axes = plt.subplots(
            numParams, 1, 
            figsize=(8, max(2.5, numParams * 1.2)), 
            squeeze=False
        )
        
        plt.subplots_adjust(hspace=0.4)
        
        for i, (paramName, optimalValue) in enumerate(boundedParams.items()):
            ax = axes[i, 0]
            
            if paramName in paramBounds:
                lowerBound, upperBound = paramBounds[paramName]
                
                # Create parameter range bar
                bar = ax.bar(
                    [0],
                    [upperBound - lowerBound],
                    bottom=[lowerBound],
                    color='lightblue',
                    alpha=0.7,
                    edgecolor='black',
                    width=0.6,
                )
                
                # Add optimal value marker
                ax.scatter([0], [optimalValue], color='red', s=100, zorder=5)
                
                # Add value label
                ax.text(
                    0.35,
                    optimalValue,
                    f'{optimalValue:.3f}',
                    ha='left',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        alpha=0.8
                    ),
                )
                
                # Set axis properties
                rangePadding = (upperBound - lowerBound) * 0.05
                ax.set_ylim(lowerBound - rangePadding, upperBound + rangePadding)
                ax.set_xlim(-0.5, 0.5)
                ax.set_xticks([])
                
                # Parameter name as y-label
                formattedName = self._formatParameterName(paramName)
                ax.set_ylabel(
                    formattedName,
                    fontweight='bold',
                    fontsize=11,
                    rotation=0,
                    ha='right',
                    va='center',
                )
                
                ax.grid(True, alpha=0.3, axis='y')
                
                # Title on first subplot
                if i == 0:
                    ax.set_title(
                        'Optimal Parameter Values within Bounds',
                        fontsize=13,
                        pad=15,
                    )
                    
        # Add legend
        if numParams > 0:
            legendElements = [
                Patch(facecolor='lightblue', alpha=0.7, label='Parameter\nRange'),
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor='red',
                    markersize=10,
                    label='Optimal\nValue',
                ),
            ]
            
            fig.legend(
                handles=legendElements,
                loc='center right',
                bbox_to_anchor=(0.95, 0.5),
                fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True,
            )
            
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(right=0.78)
        
        filename = f"{data.studyName}-parameter-positions"
        return self._savePlot(outputDir, filename)
        
    def _plotParameterSpace(self, data: StudyData, outputDir: str) -> List[str]:
        """Plot parameter space exploration from evaluation data."""
        createdFiles = []
        
        if not data.evaluationData:
            return createdFiles
            
        # Extract data
        objectives = [result['objective'] for result in data.evaluationData]
        paramNames = list(data.evaluationData[0]['parameters'].keys())
        
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
        
    def _plotParameterVsObjective(self, data: StudyData, paramNames: List[str], 
                                   objectives: List[float], outputDir: str) -> str:
        """Plot each parameter vs objective value."""
        nParams = len(paramNames)
        fig, axes = plt.subplots(
            nParams, 1, 
            figsize=(10, max(6, nParams * 2)), 
            squeeze=False
        )
        plt.subplots_adjust(hspace=0.4)
        
        for i, paramName in enumerate(paramNames):
            paramValues = [
                result['parameters'][paramName] 
                for result in data.evaluationData
            ]
            ax = axes[i, 0]
            
            # Color points by objective value
            scatter = ax.scatter(
                paramValues, objectives, 
                c=objectives, cmap='viridis', 
                alpha=0.6, s=20
            )
            
            formattedName = self._formatParameterName(paramName)
            ax.set_xlabel(formattedName)
            ax.set_ylabel('Objective Value')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title('Parameter vs Objective Value')
                plt.colorbar(scatter, ax=ax, label='Objective Value')
                
        filename = f"{data.studyName}-parameter-objective"
        return self._savePlot(outputDir, filename)
        
    def _plot2dParameterSpace(self, data: StudyData, paramNames: List[str],
                               objectives: List[float], outputDir: str) -> str:
        """Plot 2D parameter space using first two parameters."""
        param1Values = [
            result['parameters'][paramNames[0]] 
            for result in data.evaluationData
        ]
        param2Values = [
            result['parameters'][paramNames[1]] 
            for result in data.evaluationData
        ]
        
        self._setupPlot(figsize=(10, 8))
        
        scatter = plt.scatter(
            param1Values, param2Values, 
            c=objectives, cmap='viridis', 
            alpha=0.6, s=30
        )
        plt.colorbar(scatter, label='Objective Value')
        
        formattedName1 = self._formatParameterName(paramNames[0])
        formattedName2 = self._formatParameterName(paramNames[1])
        plt.xlabel(formattedName1)
        plt.ylabel(formattedName2)
        plt.title(f'Parameter Space Exploration: {formattedName1} vs {formattedName2}')
        plt.grid(True, alpha=0.3)
        
        # Highlight best point if available
        if hasattr(data, 'results') and data.results and 'bestParameters' in data.results:
            bestParams = data.results['bestParameters']
            bestX = bestParams.get(paramNames[0])
            bestY = bestParams.get(paramNames[1])
            bestObj = min(objectives)
            
            if bestX is not None and bestY is not None:
                plt.scatter([bestX], [bestY], c='red', s=100, marker='*', 
                          edgecolor='black', label=f'Best (Obj: {bestObj:.6f})')
                plt.legend()
                
        filename = f"{data.studyName}-parameter-space-2d"
        return self._savePlot(outputDir, filename)
        
    def _plotParameterSpaceFromProgress(self, data: StudyData, outputDir: str) -> List[str]:
        """Plot parameter space from progress data when evaluation data not available."""
        # This would be used for optimization runs that don't have detailed evaluation data
        # Implementation would extract parameter values from progress data if available
        # For now, return empty list as this is a lower priority feature
        return []