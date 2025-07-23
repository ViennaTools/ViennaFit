"""
Convergence plot generation for optimization and GSS results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ..base import BasePlotter, StudyData


class ConvergencePlotter(BasePlotter):
    """Generates convergence plots showing objective function evolution."""
    
    def plot(self, data: StudyData, outputDir: str) -> List[str]:
        """Generate convergence plots."""
        createdFiles = []
        
        if not data.progressData:
            return createdFiles
            
        # Plot all evaluations convergence
        if 'all' in data.progressData:
            filepath = self._plotAllEvaluationsConvergence(
                data, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        # Plot best evaluations convergence  
        if 'best' in data.progressData:
            filepath = self._plotBestEvaluationsConvergence(
                data, outputDir
            )
            if filepath:
                createdFiles.append(filepath)
                
        return createdFiles
        
    def _plotAllEvaluationsConvergence(self, data: StudyData, outputDir: str) -> str:
        """Plot convergence showing all evaluations with running best."""
        progressAll = data.progressData['all']
        
        # Extract data based on format
        if hasattr(progressAll, 'values'):  # pandas DataFrame
            evalNumbers = progressAll['evaluationNumber'].values
            objectiveValues = progressAll['objectiveValue'].values
        else:  # numpy array
            evalNumbers = np.arange(1, len(progressAll) + 1)
            objectiveValues = progressAll[:, -1]
            
        # Calculate running minimum (best so far)
        runningBest = np.minimum.accumulate(objectiveValues)
        
        self._setupPlot()
        
        # Plot all evaluations
        plt.plot(evalNumbers, objectiveValues, 'b-', linewidth=1, alpha=0.7, 
                label='All Evaluations')
        plt.scatter(evalNumbers, objectiveValues, color='blue', s=10, alpha=0.4)
        
        # Plot running best
        plt.plot(evalNumbers, runningBest, 'r-', linewidth=2, label='Best So Far')
        
        plt.xlabel('Evaluation Number')
        plt.ylabel('Objective Function Value')
        
        title = f'{data.studyType.replace("_", " ").title()} Convergence'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{data.studyName}-convergence-all"
        return self._savePlot(outputDir, filename)
        
    def _plotBestEvaluationsConvergence(self, data: StudyData, outputDir: str) -> str:
        """Plot convergence showing only best evaluations."""
        progressBest = data.progressData['best']
        
        # Extract data based on format
        if hasattr(progressBest, 'values'):  # pandas DataFrame
            evalNumbers = progressBest['evaluationNumber'].values
            objectiveValues = progressBest['objectiveValue'].values
        else:  # numpy array
            evalNumbers = np.arange(1, len(progressBest) + 1)
            objectiveValues = progressBest[:, -1]
            
        if len(evalNumbers) == 0:
            return None
            
        self._setupPlot()
        
        plt.plot(evalNumbers, objectiveValues, 'g-', linewidth=2, 
                label='Best Evaluations')
        plt.scatter(evalNumbers, objectiveValues, color='green', s=30, alpha=0.7)
        
        plt.xlabel('Evaluation Number')
        plt.ylabel('Best Objective Function Value')
        plt.title('Best Evaluations Progress')
        plt.grid(True, alpha=0.3)
        
        filename = f"{data.studyName}-convergence-best"
        return self._savePlot(outputDir, filename)