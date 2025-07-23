"""
Sensitivity analysis plot generation for GSS results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ..base import BasePlotter, StudyData


class SensitivityPlotter(BasePlotter):
    """Generates sensitivity analysis plots for GSS results."""
    
    def plot(self, data: StudyData, outputDir: str) -> List[str]:
        """Generate sensitivity analysis plots."""
        createdFiles = []
        
        if not data.sensitivityData:
            return createdFiles
            
        sensitivityResults = data.sensitivityData
        method = sensitivityResults.get('method', 'sobol')
        
        # First-order sensitivity indices
        if 'firstOrder' in sensitivityResults:
            filepath = self._plotFirstOrderIndices(
                data, sensitivityResults, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        # Total-order sensitivity indices
        if 'totalOrder' in sensitivityResults:
            filepath = self._plotTotalOrderIndices(
                data, sensitivityResults, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        # Combined comparison of first and total order
        if ('firstOrder' in sensitivityResults and 
            'totalOrder' in sensitivityResults):
            filepath = self._plotSensitivityComparison(
                data, sensitivityResults, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        # Second-order indices heatmap
        if 'secondOrder' in sensitivityResults:
            filepath = self._plotSecondOrderHeatmap(
                data, sensitivityResults, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        return createdFiles
        
    def _plotFirstOrderIndices(self, data: StudyData, sensitivityResults: Dict, 
                                outputDir: str) -> str:
        """Plot first-order sensitivity indices."""
        firstOrder = sensitivityResults['firstOrder']
        firstOrderConf = sensitivityResults.get('firstOrderConf', {})
        method = sensitivityResults.get('method', 'sobol')
        
        # Sort parameters by sensitivity value
        sortedParams = sorted(
            firstOrder.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        paramNames = [p[0] for p in sortedParams]
        values = [p[1] for p in sortedParams]
        errors = [firstOrderConf.get(name, 0) for name in paramNames]
        
        self._setupPlot()
        
        bars = plt.bar(
            paramNames, values, 
            yerr=errors, capsize=5, 
            color='skyblue', edgecolor='navy', 
            alpha=0.7
        )
        
        # Color negative bars differently
        for bar, val in zip(bars, values):
            if val < 0:
                bar.set_color('lightcoral')
                
        plt.title(f'First-Order Sensitivity Indices ({method.upper()})')
        plt.ylabel('Sensitivity Index')
        plt.xlabel('Parameters')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        filename = f"{data.studyName}-first-order-indices"
        return self._savePlot(outputDir, filename)
        
    def _plotTotalOrderIndices(self, data: StudyData, sensitivityResults: Dict,
                                outputDir: str) -> str:
        """Plot total-order sensitivity indices."""
        totalOrder = sensitivityResults['totalOrder']
        totalOrderConf = sensitivityResults.get('totalOrderConf', {})
        method = sensitivityResults.get('method', 'sobol')
        
        # Sort parameters by sensitivity value
        sortedParams = sorted(
            totalOrder.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        paramNames = [p[0] for p in sortedParams]
        values = [p[1] for p in sortedParams]
        errors = [totalOrderConf.get(name, 0) for name in paramNames]
        
        self._setupPlot()
        
        bars = plt.bar(
            paramNames, values, 
            yerr=errors, capsize=5,
            color='lightgreen', edgecolor='darkgreen', 
            alpha=0.7
        )
        
        # Color negative bars differently
        for bar, val in zip(bars, values):
            if val < 0:
                bar.set_color('lightcoral')
                
        plt.title(f'Total-Order Sensitivity Indices ({method.upper()})')
        plt.ylabel('Sensitivity Index')
        plt.xlabel('Parameters')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        filename = f"{data.studyName}-total-order-indices"
        return self._savePlot(outputDir, filename)
        
    def _plotSensitivityComparison(self, data: StudyData, sensitivityResults: Dict,
                                   outputDir: str) -> str:
        """Plot comparison of first-order and total-order indices."""
        firstOrder = sensitivityResults['firstOrder']
        totalOrder = sensitivityResults['totalOrder']
        method = sensitivityResults.get('method', 'sobol')
        
        # Get common parameters and sort by total order sensitivity
        commonParams = set(firstOrder.keys()) & set(totalOrder.keys())
        sortedParams = sorted(
            commonParams, 
            key=lambda x: abs(totalOrder[x]), 
            reverse=True
        )
        
        firstValues = [firstOrder[name] for name in sortedParams]
        totalValues = [totalOrder[name] for name in sortedParams]
        
        x = np.arange(len(sortedParams))
        width = 0.35
        
        self._setupPlot(figsize=(12, 6))
        
        plt.bar(x - width/2, firstValues, width, label='First-Order', 
               color='skyblue', edgecolor='navy', alpha=0.7)
        plt.bar(x + width/2, totalValues, width, label='Total-Order',
               color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        
        plt.title(f'Sensitivity Indices Comparison ({method.upper()})')
        plt.ylabel('Sensitivity Index')
        plt.xlabel('Parameters')
        plt.xticks(x, sortedParams, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        filename = f"{data.studyName}-sensitivity-comparison"
        return self._savePlot(outputDir, filename)
        
    def _plotSecondOrderHeatmap(self, data: StudyData, sensitivityResults: Dict,
                                 outputDir: str) -> Optional[str]:
        """Plot second-order sensitivity indices as a heatmap."""
        secondOrder = sensitivityResults['secondOrder']
        method = sensitivityResults.get('method', 'sobol')
        
        if not secondOrder:
            return None
            
        # Get parameter names from first-order results or metadata
        if 'firstOrder' in sensitivityResults:
            paramNames = list(sensitivityResults['firstOrder'].keys())
        elif data.metadata and 'parameterNames' in data.metadata:
            paramNames = data.metadata['parameterNames']
        else:
            return None
            
        nParams = len(paramNames)
        matrix = np.zeros((nParams, nParams))
        
        # Fill matrix with second-order indices
        for pairKey, value in secondOrder.items():
            if '_' in pairKey:
                p1, p2 = pairKey.split('_', 1)
                if p1 in paramNames and p2 in paramNames:
                    i, j = paramNames.index(p1), paramNames.index(p2)
                    matrix[i, j] = value
                    matrix[j, i] = value  # Make symmetric
                    
        self._setupPlot(figsize=(8, 6))
        
        im = plt.imshow(matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, label='Second-Order Sensitivity Index')
        
        plt.xticks(range(nParams), paramNames, rotation=45, ha='right')
        plt.yticks(range(nParams), paramNames)
        plt.title(f'Second-Order Sensitivity Indices Heatmap ({method.upper()})')
        
        # Add value annotations
        for i in range(nParams):
            for j in range(nParams):
                if i != j:  # Don't annotate diagonal
                    color = 'white' if abs(matrix[i,j]) > 0.5 else 'black'
                    plt.text(j, i, f'{matrix[i,j]:.3f}', 
                           ha='center', va='center',
                           color=color, fontsize=8)
                           
        filename = f"{data.studyName}-second-order-heatmap"
        return self._savePlot(outputDir, filename)