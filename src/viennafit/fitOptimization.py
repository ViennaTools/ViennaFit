from .fitProject import Project
from .fitOptimizerWrapper import OptimizerWrapper
from .fitStudy import Study
import viennaps2d as vps
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
import ast
from typing import Dict, List, Tuple


class Optimization(Study):
    def __init__(self, name: str, project: Project):
        super().__init__(name, project, "optimizationRuns")
        self.optimizer = "dlib"  # Default optimizer

    def setParameterNames(self, paramNames: List[str]):
        """Specifies names of parameters that will be used in optimization"""
        self.parameterNames = paramNames
        return self

    def setFixedParameters(self, fixedParams: Dict[str, float]):
        """
        Set multiple parameters as fixed with specific values

        Args:
            fixedParams: Dictionary mapping parameter names to fixed values
        """
        if self.parameterNames is None:
            raise ValueError(
                "Parameter names must be set before defining fixed parameters"
            )
        for name, value in fixedParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names"
                )
            if name in self.variableParameters:
                raise ValueError(f"Parameter '{name}' is already set as variable")
            self.fixedParameters[name] = value
        return self

    def setVariableParameters(self, varParams: Dict[str, Tuple[float, float]]):
        """
        Set multiple parameters as variable with ranges

        Args:
            varParams: Dictionary mapping parameter names to tuples of (lowerBound, upperBound)
        """
        if not self.parameterNames:
            raise ValueError(
                "Parameter names must be set before defining variable parameters"
            )
        for name, (lowerBound, upperBound) in varParams.items():
            if name not in self.parameterNames:
                raise ValueError(
                    f"Parameter '{name}' is not defined in parameter names. "
                    f"The current parameter names are: {self.parameterNames}"
                )
            if name in self.fixedParameters:
                raise ValueError(f"Parameter '{name}' is already set as fixed")
            self.variableParameters[name] = (lowerBound, upperBound)
        return self

    def getVariableParameterList(self):
        """Get list of variable parameters for optimization algorithms"""
        return self.variableParameters.keys()

    def getVariableBounds(self):
        """Get bounds for variable parameters as lists"""
        varParams = self.getVariableParameterList()
        lowerBounds = [p.lowerBound for p in varParams]
        upperBounds = [p.upperBound for p in varParams]
        return lowerBounds, upperBounds

    def saveResults(self, filename: str = "results.json"):
        """Save results to file"""
        filepath = os.path.join(self.runDir, filename)

        result = {
            "bestScore": self.bestScore,
            "bestEvaluation#": self.bestEvaluationNumber,
            "bestParameters": self.bestParameters,
            "fixedParameters": self.fixedParameters,
            "variableParameters": self.variableParameters,
            "optimizer": self.optimizer,
            "numEvaluations": self.numEvaluations,
        }

        with open(filepath, "w") as f:
            json.dump(result, f, indent=4)

        bestDomainFile = f"{self.name}-{self.bestEvaluationNumber}.vtp"
        bestDomainPath = os.path.join(self.runDir, "progress", bestDomainFile)
        if os.path.exists(bestDomainPath):
            projectDomainDir = os.path.join(
                self.project.projectAbsPath, "domains", "optimalDomains"
            )
            os.makedirs(projectDomainDir, exist_ok=True)
            targetPath = os.path.join(projectDomainDir, bestDomainFile)
            shutil.copy2(bestDomainPath, targetPath)
            print(f"Best domain copied to {targetPath}")
        else:
            print(f"Best domain file {bestDomainFile} not found in progress directory")

        print(f"Results saved to {filepath}")

    def _parseProgressFile(self, file_path: str):
        """Parse progress file containing Python list strings into numpy array"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data_rows = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        # Parse the Python list string
                        data_list = ast.literal_eval(line)
                        if isinstance(data_list, list):
                            # Convert strings to floats
                            data_row = [float(x) for x in data_list]
                            data_rows.append(data_row)
                    except (ValueError, SyntaxError):
                        # Skip malformed lines
                        continue
            
            if data_rows:
                # Convert to numpy array, handling variable column counts
                max_cols = max(len(row) for row in data_rows)
                # Pad shorter rows with NaN if needed
                padded_rows = []
                for row in data_rows:
                    if len(row) < max_cols:
                        row.extend([np.nan] * (max_cols - len(row)))
                    padded_rows.append(row)
                
                return np.array(padded_rows)
            else:
                return np.array([])
                
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return np.array([])

    def saveVisualizationPlots(self):
        """Save matplotlib visualization plots after optimization completion"""
        if not self.applied or not self.bestParameters:
            print("No optimization results to visualize")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.runDir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Variable parameter position plots
        if self.variableParameters:
            fig, axes = plt.subplots(len(self.variableParameters), 1, 
                                   figsize=(10, 2*len(self.variableParameters)))
            if len(self.variableParameters) == 1:
                axes = [axes]
            
            for i, (param_name, (lower_bound, upper_bound)) in enumerate(self.variableParameters.items()):
                if param_name in self.bestParameters:
                    optimal_value = self.bestParameters[param_name]
                    
                    # Create range visualization
                    x_range = np.linspace(lower_bound, upper_bound, 100)
                    y_line = np.ones_like(x_range) * 0.5
                    
                    axes[i].plot(x_range, y_line, 'k-', linewidth=2, label='Parameter range')
                    axes[i].scatter([optimal_value], [0.5], color='red', s=100, 
                                  label=f'Optimum: {optimal_value:.4f}', zorder=5)
                    axes[i].set_xlim(lower_bound, upper_bound)
                    axes[i].set_ylim(0, 1)
                    axes[i].set_xlabel(f'{param_name}')
                    axes[i].set_ylabel('Position')
                    axes[i].set_title(f'Optimal {param_name} within bounds')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    
                    # Remove y-axis ticks as they're not meaningful
                    axes[i].set_yticks([])
            
            plt.tight_layout()
            param_plot_path = os.path.join(plots_dir, f"{self.name}-parameter-positions.png")
            plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Parameter position plots saved to {param_plot_path}")
        
        # 2. Convergence plot for progressAll.txt
        progress_all_file = os.path.join(self.runDir, "progressAll.txt")
        if os.path.exists(progress_all_file):
            data = self._parseProgressFile(progress_all_file)
            if data.size > 0:
                plt.figure(figsize=(10, 6))
                # Use first column as evaluation number and last column as objective value
                eval_numbers = np.arange(1, len(data) + 1)  # Create evaluation numbers
                objective_values = data[:, -1]  # Last column is typically objective value
                
                plt.plot(eval_numbers, objective_values, 'b-', linewidth=2)
                plt.scatter(eval_numbers, objective_values, color='blue', s=20, alpha=0.6)
                plt.xlabel('Evaluation Number')
                plt.ylabel('Objective Function Value')
                plt.title('Convergence History (All Evaluations)')
                plt.grid(True, alpha=0.3)
                
                convergence_all_path = os.path.join(plots_dir, f"{self.name}-convergence-all.png")
                plt.savefig(convergence_all_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Convergence plot (all evaluations) saved to {convergence_all_path}")
            else:
                print("No valid data found in progressAll.txt")
        
        # 3. Convergence plot for progress.txt
        progress_file = os.path.join(self.runDir, "progress.txt")
        if os.path.exists(progress_file):
            data = self._parseProgressFile(progress_file)
            if data.size > 0:
                plt.figure(figsize=(10, 6))
                # Use first column as evaluation number and last column as objective value
                eval_numbers = np.arange(1, len(data) + 1)  # Create evaluation numbers
                objective_values = data[:, -1]  # Last column is typically objective value
                
                plt.plot(eval_numbers, objective_values, 'g-', linewidth=2)
                plt.scatter(eval_numbers, objective_values, color='green', s=20, alpha=0.6)
                plt.xlabel('Evaluation Number')
                plt.ylabel('Best Objective Function Value')
                plt.title('Convergence History (Best Values)')
                plt.grid(True, alpha=0.3)
                
                convergence_best_path = os.path.join(plots_dir, f"{self.name}-convergence-best.png")
                plt.savefig(convergence_best_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Convergence plot (best values) saved to {convergence_best_path}")
            else:
                print("No valid data found in progress.txt")

    def saveStartingConfiguration(self):
        """Save the starting configuration of the optimization"""
        if not self.applied:
            raise RuntimeError(
                "Optimization must be applied before saving configuration"
            )

        config = {
            "name": self.name,
            "parameterNames": self.parameterNames,
            "fixedParameters": self.fixedParameters,
            "variableParameters": self.variableParameters,
            "optimizer": self.optimizer,
            "numEvaluations": self.numEvaluations,
        }

        configFile = os.path.join(
            self.runDir, self.name + "-startingConfiguration.json"
        )
        with open(configFile, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Starting configuration saved to {configFile}")

    def setOptimizer(self, optimizer: str):
        """Set the optimizer to be used"""
        self.optimizer = optimizer
        return self

    def apply(
        self,
        numEvaluations: int = 100,
        saveAllEvaluations: bool = False,
        saveVisualization: bool = True,
    ):
        """Apply the optimization."""
        if not self.applied:
            self.validate()
            self.saveVisualization = saveVisualization
            self.saveAllEvaluations = saveAllEvaluations
            self.applied = True
            self.evalCounter = 0
            self.numEvaluations = numEvaluations
            self.saveStartingConfiguration()
        else:
            print("Optimization has already been applied.")
            return

        # Create optimizer wrapper
        optimizer = OptimizerWrapper.create(self.optimizer, self)

        try:
            # Run optimization
            result = optimizer.optimize(self.numEvaluations)

            # Save results
            if result["success"]:
                if self.bestParameters is None:
                    self.bestParameters = {}
                self.bestParameters.update(result["x"])
                self.bestScore = result["fun"]

                print(f"Optimization completed successfully:")
                print(f"  Function evaluations: {result['nfev']}")
                print(f"  Best score: {result['fun']:.6f}")
                print("  Best parameters:")
                for name, value in result["x"].items():
                    print(f"    {name}: {value:.6f}")
                print(f" Best evaluation #: {self.bestEvaluationNumber}")

                # Save final results
                self.saveResults(self.name + "-final-results.json")
                
                # Save visualization plots if requested
                if self.saveVisualization:
                    self.saveVisualizationPlots()

            else:
                print("Optimization failed to converge")

        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            raise
