"""
Multi-run comparison plot generation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ..base import BasePlotter, StudyData
from ..loaders import ResultsLoader


class ComparisonPlotter(BasePlotter):
    """Generates comparison plots between multiple runs."""
    
    def __init__(self, run_dirs: List[str], config=None):
        """
        Initialize comparison plotter with multiple run directories.
        
        Args:
            run_dirs: List of run directory paths to compare
            config: Plot configuration
        """
        super().__init__(config)
        self.run_dirs = [os.path.abspath(d) for d in run_dirs]
        self._study_data = []
        
        # Load data for all runs
        for run_dir in self.run_dirs:
            try:
                loader = ResultsLoader(run_dir)
                data = loader.load_study_data()
                self._study_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load data from {run_dir}: {e}")
                
    def plot(self, data: StudyData, output_dir: str) -> List[str]:
        """
        Generate comparison plots.
        
        Note: For comparison plots, the 'data' parameter is not used as we work with
        multiple datasets loaded in __init__.
        """
        created_files = []
        
        if len(self._study_data) < 2:
            print("Warning: Need at least 2 runs for comparison plots")
            return created_files
            
        # Convergence comparison
        filepath = self._plot_convergence_comparison(output_dir)
        if filepath:
            created_files.append(filepath)
            
        # Best results comparison (for optimization runs)
        opt_studies = [d for d in self._study_data if d.study_type == 'optimization']
        if len(opt_studies) >= 2:
            filepath = self._plot_best_results_comparison(opt_studies, output_dir)
            if filepath:
                created_files.append(filepath)
                
        return created_files
        
    def _plot_convergence_comparison(self, output_dir: str) -> Optional[str]:
        """Plot convergence comparison across multiple runs."""
        self._setup_plot(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self._study_data)))
        
        for i, (data, color) in enumerate(zip(self._study_data, colors)):
            if not data.progress_data or 'all' not in data.progress_data:
                continue
                
            progress_all = data.progress_data['all']
            
            # Extract data based on format
            if hasattr(progress_all, 'values'):  # pandas DataFrame
                eval_numbers = progress_all['evaluationNumber'].values
                objective_values = progress_all['objectiveValue'].values
            else:  # numpy array
                eval_numbers = np.arange(1, len(progress_all) + 1)
                objective_values = progress_all[:, -1]
                
            # Calculate running best
            running_best = np.minimum.accumulate(objective_values)
            
            plt.plot(eval_numbers, running_best, 
                    color=color, linewidth=2, 
                    label=f'{data.study_name}')
                    
        plt.xlabel('Evaluation Number')
        plt.ylabel('Best Objective Function Value')
        plt.title('Convergence Comparison Across Runs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        filename = "convergence-comparison"
        return self._save_plot(output_dir, filename)
        
    def _plot_best_results_comparison(self, opt_studies: List[StudyData], 
                                    output_dir: str) -> Optional[str]:
        """Plot comparison of best results from optimization runs."""
        best_scores = []
        study_names = []
        
        for data in opt_studies:
            if data.results and 'bestScore' in data.results:
                best_scores.append(data.results['bestScore'])
                study_names.append(data.study_name)
            elif hasattr(data, 'progress_data') and data.progress_data:
                # Extract best score from progress data
                if 'best' in data.progress_data:
                    progress_best = data.progress_data['best']
                    if hasattr(progress_best, 'values'):  # pandas DataFrame
                        best_score = progress_best['objectiveValue'].min()
                    else:  # numpy array
                        best_score = progress_best[:, -1].min()
                    best_scores.append(best_score)
                    study_names.append(data.study_name)
                    
        if len(best_scores) < 2:
            return None
            
        self._setup_plot(figsize=(10, 6))
        
        # Sort by best score
        sorted_data = sorted(zip(study_names, best_scores), key=lambda x: x[1])
        sorted_names, sorted_scores = zip(*sorted_data)
        
        bars = plt.bar(sorted_names, sorted_scores, 
                      color='lightblue', edgecolor='navy', alpha=0.7)
        
        # Highlight the best run
        bars[0].set_color('lightgreen')
        
        plt.xlabel('Study Runs')
        plt.ylabel('Best Objective Function Value')
        plt.title('Best Results Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.6f}',
                    ha='center', va='bottom', fontsize=9)
        
        filename = "best-results-comparison"
        return self._save_plot(output_dir, filename)