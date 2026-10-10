# Tutorial 2: Custom Evaluation

Evaluate parameter sets of your own choice with the process sequence of a finished optimization run: grids around the optimum and repeated runs of the optimum.

This tutorial walks through `examples/2-optimization/customEvaluatorExample.py` and the two scripts in `examples/3-custom-evaluations/`. Every code block below is taken from those scripts.

**Estimated time:** 10-15 minutes

## What You'll Learn

- Load an optimization run into a `CustomEvaluator`
- Evaluate a grid of fixed parameter values
- Evaluate a grid placed relative to the optimum
- Measure how much the result varies between identical runs

## Prerequisites

- Completed [Tutorial 1](tutorial-1-basic-optimization.md), so that the run `run1` exists in `projects/exampleProject`

## Why Evaluate by Hand?

An optimization returns one best parameter set. It does not tell you how sharp that optimum is, whether other combinations fit nearly as well, or how much of the remaining distance is simulation noise. The `CustomEvaluator` answers these questions by running the same process sequence on parameter values that you pick.

## Step 1: Load the Optimization Run

All three scripts start the same way:

```python
import viennafit as fit
import os

# Load the project
p1 = fit.Project()
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectToLoad = os.path.abspath(
    os.path.join(scriptDir, "../../projects/exampleProject")
)
p1.load(projectToLoad)

# Create a custom evaluator instance
evaluator = fit.CustomEvaluator(p1)

# Load optimization results from a previous run
evaluator.loadOptimizationRun("run1")
```

`loadOptimizationRun` loads the process sequence stored in the run folder and the best parameters of that run. Any parameter you do not vary afterwards stays at its optimal value.

```python
optimalParams = evaluator.getOptimalParameters()
print("Optimal parameters from optimization run:")
for paramName, value in optimalParams.items():
    print(f"  {paramName}: {value}")
```

!!! note
    If you ran the optimization more than once, the later runs are called `run1_1`, `run1_2`, and so on. Pass the name of the run you want to build on.

## Step 2: A Grid of Fixed Values

Script: `examples/2-optimization/customEvaluatorExample.py`

```python
# Set distance metric (same as used in optimization)
evaluator.setDistanceMetric("CCH")
evaluator.setAdditionalMetrics(["CSF"])

# This will sweep neutralRate and ionRate over the region around the optimum
# (5 x 4 = 20 evaluations)
variableValues = {
    "neutralRate": [70.0, 85.0, 100.0, 115.0, 130.0],
    "ionRate": [1.0, 2.0, 3.0, 4.0],
}

evaluator.setVariableValues(variableValues)
```

`setVariableValues` takes a list of values per parameter and evaluates every combination. The two sticking and angular parameters are not listed, so they stay at their optimum.

`setAdditionalMetrics` records further metrics next to the primary one. `CSF` compares the two level sets on their sparse field, which gives a second opinion on the same pair of surfaces.

```python
results = evaluator.apply(evaluationName="parameterSweep1", saveComparison=True)

bestResult = evaluator.getBestResult()
if bestResult:
    print(f"\nBest result from grid evaluation:")
    print(f"  Objective value: {bestResult['objectiveValue']:.6f}")
    print(f"  Parameters:")
    for paramName, value in bestResult["parameters"].items():
        print(f"    {paramName}: {value:.6f}")

# Save a detailed report
evaluator.saveGridReport()
```

Run it:

```bash
cd examples/2-optimization
python customEvaluatorExample.py
```

The 20 evaluations take about a minute. The results are written to `projects/exampleProject/customEvaluations/parameterSweep1/`:

| File | Content |
|---|---|
| `grid_results_summary.csv` | One row per evaluation: parameter values, metric values, execution time |
| `grid_evaluation_report.json` | The same, with the configuration of the evaluation |
| `neutralRate-100_ionRate-2-result.vtp` etc. | Simulated surface of each grid point (with `saveComparison=True`) |
| `parameterSweep1-processSequence.py` | Copy of the process sequence that was used |

### Reading the grid

The Chamfer distance in nm over the grid, from one run (your numbers will differ):

| `neutralRate` | `ionRate` = 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 70 | 21.0 | 15.8 | 10.8 | 7.4 |
| 85 | 12.7 | 7.5 | 3.8 | 6.3 |
| 100 | 5.1 | **3.1** | 7.5 | 12.9 |
| 115 | 5.8 | 10.5 | 15.4 | 20.8 |
| 130 | 13.4 | 18.8 | 23.5 | 28.9 |

The good fits lie on a diagonal: less neutral deposition can be made up for by more ion deposition, and the other way round. This is a trade-off between the two rates, and it is typical for process models. It means the total amount of deposited material is pinned down much better than the split between neutral and ion.

The valley is still narrow: one step off the diagonal the distance doubles, so the fit does constrain both rates.

## Step 3: A Grid Relative to the Optimum

Script: `examples/3-custom-evaluations/grid-customEvaluator.py`

Fixed values have to be edited whenever the optimum moves. The values can also be derived from the optimum itself:

```python
optimalParams = evaluator.getOptimalParameters()

# 4 values each, uniformly spaced within ±20% of the optimum
factors = [0.8, 0.933, 1.067, 1.2]
variableValues = {
    "neutralRate": [optimalParams["neutralRate"] * f for f in factors],
    "ionRate": [optimalParams["ionRate"] * f for f in factors],
}

evaluator.setVariableValues(variableValues)

results = evaluator.apply(evaluationName="optimumGrid", saveComparison=True)
```

```bash
cd ../3-custom-evaluations
python grid-customEvaluator.py
```

This gives a 4 x 4 grid (16 evaluations, under a minute) that is centred on the optimum and works for any run. The factors are chosen so that the optimum itself is not on the grid. If the best grid point is clearly better than the optimization result, the optimization had not converged.

The output goes to `customEvaluations/optimumGrid/`, in the same format as before.

## Step 4: Repeatability

Script: `examples/3-custom-evaluations/repeated-customEvaluator.py`

The simulation uses Monte Carlo ray tracing, so the same parameters do not give exactly the same surface twice. To see how large this effect is, evaluate the optimum several times:

```python
bestParams = evaluator.getOptimalParameters()

# This runs the same parameters multiple times to assess reproducibility
evaluator.setConstantParametersWithRepeats(bestParams, numRepeats=10)

evaluator.setDistanceMetric("CCH")

results = evaluator.apply(evaluationName="repeatedEvaluation", saveComparison=True)
```

```bash
python repeated-customEvaluator.py
```

Ten runs take about half a minute. The output goes to `customEvaluations/repeatedEvaluation/`. Besides the individual results, ViennaFit compares the ten result surfaces with each other and writes the mean, standard deviation, minimum and maximum of these pairwise distances to `repeatability_stats.json`.

### What the noise tells you

The spread of the objective value over the ten runs is the noise floor of your setup:

- Differences between two parameter sets that are smaller than this spread mean nothing.
- An optimization cannot be expected to converge more tightly than this.
- If the noise is too large for your purpose, raise `raysPerPoint` in the process sequence. The evaluations get slower in return.

## Key Takeaways

- `loadOptimizationRun` gives you the process sequence and the optimum of a run.
- `setVariableValues` evaluates all combinations of the values you list; unlisted parameters stay at the optimum.
- A grid shows trade-offs between parameters that a single optimum hides.
- `setConstantParametersWithRepeats` measures the noise floor of the simulation.

## Next Steps

- [Tutorial 3: Sensitivity Analysis](tutorial-3-sensitivity-analysis.md) - quantify which parameters matter
- [Core Concepts](../getting-started/concepts.md) - background on metrics and evaluators
