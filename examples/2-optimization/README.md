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
- Evaluates a 5 x 4 grid of `neutralRate` and `ionRate` around the optimum
- Generates detailed reports and visualizations

### optional-finalizeOptimization.py
- Only needed if an optimization run was stopped before it completed
- Writes the final results, best domain and plots for the incomplete run

## Prerequisites

Before running optimization examples:
1. Complete the setup examples (project initialization and domain creation)
2. Ensure the project has been properly initialized with initial and target domains

## Usage

1. Run `basicOptimization.py` first to perform the optimization
2. After optimization completes, run `customEvaluatorExample.py` to explore parameter variations
3. Check the project's `optimizationRuns/` and `customEvaluations/` folders for results

## Distance Metrics

The examples use CCH (Chamfer distance) as the primary metric and track one additional metric:
- **CCH (Compare Chamfer)**: Primary metric - bidirectional distance metric.
- **CA (Compare Area)**: Additional metric in `basicOptimization.py` - tracked separately.
- **CSF (Compare Sparse Field)**: Additional metric in `customEvaluatorExample.py` - tracked separately.

The optimization is driven by the Chamfer distance only; the additional metrics are recorded for analysis.
