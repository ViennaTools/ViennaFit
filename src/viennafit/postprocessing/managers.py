"""
Postprocessing manager classes for optimization and GSS studies.
"""

from typing import List, Optional
from .base import BasePostprocessor, PlotConfig, StudyData
from .loaders import ResultsLoader
from .plotters import ConvergencePlotter, ParameterPlotter, ParameterProgressionPlotter, SensitivityPlotter


class OptimizationPostprocessor(BasePostprocessor):
    """Postprocessor specialized for optimization study results."""
    
    def loadData(self) -> StudyData:
        """Load optimization study data."""
        loader = ResultsLoader(self.runDir)
        return loader.loadStudyData()
        
    def getAvailablePlotters(self) -> List:
        """Get plotters applicable to optimization studies."""
        plotters = [
            ConvergencePlotter(self.config),
            ParameterPlotter(self.config),
            ParameterProgressionPlotter(self.config),
        ]
        return plotters
        
    def generateAllPlots(self) -> dict:
        """Generate all standard plots for optimization results."""
        return self.generatePlots()
        
    def generateSummaryReport(self) -> str:
        """Generate a summary report of the optimization results."""
        data = self.loadData()
        
        summaryLines = [
            f"# Optimization Results Summary: {data.studyName}",
            "",
            f"**Study Type**: {data.studyType}",
            f"**Run Directory**: {data.runDir}",
            "",
        ]
        
        # Add metadata information
        if data.metadata:
            summaryLines.extend([
                "## Configuration",
                f"- Optimizer: {data.metadata.get('optimizer', 'Unknown')}",
                f"- Parameter Names: {', '.join(data.metadata.get('parameterNames', []))}",
                "",
            ])
            
        # Add results information
        if data.results:
            summaryLines.extend([
                "## Results",
                f"- Best Score: {data.results.get('bestScore', 'N/A')}",
                f"- Best Evaluation: #{data.results.get('bestEvaluation#', 'N/A')}",
                f"- Total Evaluations: {data.results.get('numEvaluations', 'N/A')}",
                "",
            ])
            
            if 'bestParameters' in data.results:
                summaryLines.append("### Best Parameters:")
                for name, value in data.results['bestParameters'].items():
                    summaryLines.append(f"- {name}: {value:.6f}")
                summaryLines.append("")
                
        # Add convergence information
        if data.progressData:
            summaryLines.extend([
                "## Progress Summary",
            ])
            
            # Count evaluations
            if 'all' in data.progressData:
                allData = data.progressData['all']
                if hasattr(allData, 'values'):
                    totalEvals = len(allData)
                else:
                    totalEvals = len(allData)
                summaryLines.append(f"- Total Evaluations Recorded: {totalEvals}")
                
            if 'best' in data.progressData:
                bestData = data.progressData['best']
                if hasattr(bestData, 'values'):
                    improvements = len(bestData)
                else:
                    improvements = len(bestData)
                summaryLines.append(f"- Number of Improvements: {improvements}")
                
        return "\n".join(summaryLines)


class GSSPostprocessor(BasePostprocessor):
    """Postprocessor specialized for Global Sensitivity Study results."""
    
    def loadData(self) -> StudyData:
        """Load GSS study data."""
        loader = ResultsLoader(self.runDir)
        return loader.loadStudyData()
        
    def getAvailablePlotters(self) -> List:
        """Get plotters applicable to GSS studies."""
        plotters = [
            ConvergencePlotter(self.config),
            ParameterPlotter(self.config),
            SensitivityPlotter(self.config),
        ]
        return plotters
        
    def generateAllPlots(self) -> dict:
        """Generate all standard plots for GSS results."""
        return self.generatePlots()
        
    def generateSensitivitySummary(self) -> str:
        """Generate a summary report of the sensitivity analysis results."""
        data = self.loadData()
        
        summaryLines = [
            f"# Global Sensitivity Study Summary: {data.studyName}",
            "",
            f"**Study Type**: {data.studyType}",
            f"**Run Directory**: {data.runDir}",
            "",
        ]
        
        # Add metadata information
        if data.metadata:
            summaryLines.extend([
                "## Configuration",
                f"- Method: Global Sensitivity Analysis",
                f"- Parameter Names: {', '.join(data.metadata.get('parameterNames', []))}",
                "",
            ])
            
        # Add sensitivity analysis results
        if data.sensitivityData:
            sensData = data.sensitivityData
            method = sensData.get('method', 'unknown')
            
            summaryLines.extend([
                f"## Sensitivity Analysis Results ({method.upper()})",
                "",
            ])
            
            # First-order indices
            if 'firstOrder' in sensData:
                firstOrder = sensData['firstOrder']
                sortedFirst = sorted(
                    firstOrder.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                summaryLines.extend([
                    "### First-Order Sensitivity Indices:",
                ])
                for name, value in sortedFirst[:5]:  # Top 5
                    summaryLines.append(f"- {name}: {value:.4f}")
                summaryLines.append("")
                
            # Total-order indices
            if 'totalOrder' in sensData:
                totalOrder = sensData['totalOrder']
                sortedTotal = sorted(
                    totalOrder.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                summaryLines.extend([
                    "### Total-Order Sensitivity Indices:",
                ])
                for name, value in sortedTotal[:5]:  # Top 5
                    summaryLines.append(f"- {name}: {value:.4f}")
                summaryLines.append("")
                
        # Add evaluation summary
        if data.evaluationData:
            totalEvaluations = len(data.evaluationData)
            objectives = [result['objective'] for result in data.evaluationData]
            bestObj = min(objectives)
            meanObj = sum(objectives) / len(objectives)
            
            summaryLines.extend([
                "## Evaluation Summary",
                f"- Total Evaluations: {totalEvaluations}",
                f"- Best Objective Value: {bestObj:.6f}",
                f"- Mean Objective Value: {meanObj:.6f}",
                "",
            ])
            
        return "\n".join(summaryLines)