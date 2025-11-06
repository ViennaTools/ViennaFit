# Optimization Examples

This folder contains examples for parameter optimization using ViennaFit.

## Files

### basicOptimization.py
- Basic parameter optimization example
- Defines a process sequence with multiple parameters
- Shows how to set parameter bounds and run optimization
- Demonstrates primary and additional metrics configuration
- Saves results for use in sensitivity analysis

### customEvaluatorExample.py
- Advanced parameter evaluation using CustomEvaluator
- Loads results from previous optimization runs
- Evaluates parameter grids around optimal values
- Generates detailed reports and visualizations

## Prerequisites

Before running optimization examples:
1. Complete the setup examples (project initialization and domain creation)
2. Ensure the project has been properly initialized with initial and target domains

## Usage

1. Run `basicOptimization.py` first to perform the optimization
2. After optimization completes, run `customEvaluatorExample.py` to explore parameter variations
3. Check the project's `optimizationRuns/` and `customEvaluations/` folders for results

## Distance Metrics

The examples use CCH (Chamfer distance) as the primary metric with CSF as an additional metric:
- **CCH (Compare Chamfer)**: Primary metric - bidirectional distance metric.
- **CSF (Compare Sparse Field)**: Additional metric - tracked separately.

This configuration optimizes based on Chamfer distance while also tracking sparse field comparison for comprehensive analysis.
