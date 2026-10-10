# Tutorial 3: Sensitivity Analysis

Find out how strongly the fit depends on each parameter, with a local study around the optimum, simple parameter sweeps, and a global study over the full parameter ranges.

This tutorial walks through the three scripts in `examples/sensitivityAnalysis/`. Every code block below is taken from those scripts.

**Estimated time:** 15-20 minutes

## What You'll Learn

- Run a local sensitivity study around the optimum
- Run one-parameter and two-parameter sweeps
- Run a global (Sobol) sensitivity study
- Decide which of the three fits your question

## Prerequisites

- Completed [Tutorial 1](tutorial-1-basic-optimization.md), so that the run `run1` exists in `projects/exampleProject`
- [Tutorial 2](tutorial-2-custom-evaluation.md) is helpful, since the parameter sweeps use the `CustomEvaluator` introduced there

## Which Study for Which Question?

| Study | Question it answers | Cost in this example |
|---|---|---|
| Local sensitivity | How does the fit change when I move one parameter away from the optimum? | 14 evaluations |
| Parameter sweeps | What does the objective look like along one or two parameters I choose? | 6 + 9 evaluations |
| Global sensitivity | Over the whole allowed range, which parameters drive the result? | 128 evaluations |

All three vary `neutralRate` and `ionRate` and keep the other parameters at their optimum.

## Common Setup

The two sensitivity studies start by loading the project, the process sequence of `run1` and its optimum:

```python
import viennafit as fit
import viennaps as vps
import json
import os

# Load the project
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../projects/exampleProject",
)
p1.load(projectToLoad)

# Use the process sequence and the optimum from a previous optimization run
runDir = os.path.abspath(os.path.join(p1.projectPath, "optimizationRuns", "run1"))
processPath = os.path.join(runDir, "run1-processSequence.py")
with open(os.path.join(runDir, "run1-final-results.json")) as f:
    optimum = json.load(f)["bestParameters"]
```

Every optimization run keeps a copy of its process sequence (`run1-processSequence.py`), so a study can reuse exactly the model that was fitted.

## Part 1: Local Sensitivity

Script: `examples/sensitivityAnalysis/localSensitivityStudy.py`

A local study moves one parameter at a time through a range while all others stay put.

```python
lss1 = fit.LocalSensitivityStudy("lss1", p1)

lss1.loadProcessSequence(processPath)

# Declare all parameters used by the process sequence
lss1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Keep the remaining parameters fixed at their optimal values
lss1.setFixedParameters(
    {name: optimum[name] for name in ["ionEnergy", "neutralStickP", "ionPowerCosine"]}
)

# Set variable parameters as (lowerBound, pointOfInterest, upperBound):
# each one is varied on its own around the optimum, within the optimization bounds
lss1.setParameterSensitivityRanges(
    {
        "neutralRate": (1.0, optimum["neutralRate"], 150.0),
        "ionRate": (0.1, optimum["ionRate"], 10.0),
    }
)

# Same distance metric as used in the optimization
lss1.setDistanceMetric("CCH")

lss1.validate()

lss1.apply()
```

The points to note:

- **`setParameterSensitivityRanges`** takes `(lowerBound, pointOfInterest, upperBound)` per parameter. The point of interest is the optimum; the bounds are the ones used in the optimization.
- By default each parameter is evaluated at 7 points across its range. Pass `nEval=(n1, n2)` with one entry per variable parameter to change that.
- **`validate()`** checks the configuration before any simulation is started.

Run it:

```bash
cd examples/sensitivityAnalysis
python localSensitivityStudy.py
```

The 14 evaluations take under a minute. The output goes to `projects/exampleProject/locSensStudies/lss1/`, with the individual evaluations in `evaluations/` and the summary in `sensitivity_results.json`.

### Reading the result

For each parameter you get the objective value along its range. Look at the shape:

- **A sharp minimum at the optimum**: the parameter is well determined by the target.
- **A flat curve**: the target says little about this parameter. Its fitted value should not be over-interpreted, and it is a candidate for being fixed.
- **A minimum away from the point of interest**: the optimization had not converged for this parameter.

Keep in mind that a local study only moves along the axes. The trade-off between `neutralRate` and `ionRate` seen in the grid of Tutorial 2 runs diagonally and does not show up here.

## Part 2: Parameter Sweeps

Script: `examples/sensitivityAnalysis/parameterStudies.py`

Sometimes a plain sweep with hand-picked values is all you need. This script uses the `CustomEvaluator` for two such studies.

### One parameter

```python
evaluator = fit.CustomEvaluator(p1)

# Load optimization results from a previous run
evaluator.loadOptimizationRun("run1")
evaluator.setDistanceMetric("CCH")

# Define a simple parameter sweep for one parameter
singleParamValues = {
    "neutralRate": [40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
}

evaluator.setVariableValues(singleParamValues)
results = evaluator.apply(evaluationName="singleParameterSweep", saveComparison=False)

print(f"Evaluated {len(results)} parameter combinations")
bestResult = evaluator.getBestResult()
if bestResult:
    print(f"Best objective value: {bestResult['objectiveValue']:.6f}")
    print(f"Best neutralRate: {bestResult['parameters']['neutralRate']:.2f}")
```

### Two parameters

```python
evaluator2 = fit.CustomEvaluator(p1)
evaluator2.loadOptimizationRun("run1")
evaluator2.setDistanceMetric("CCH")

# Study interaction between two parameters around their optimal values (±20%)
optimalParams = evaluator2.getOptimalParameters()
factors = [0.8, 1.0, 1.2]
twoParamValues = {
    "neutralRate": [optimalParams["neutralRate"] * f for f in factors],
    "ionRate": [optimalParams["ionRate"] * f for f in factors],
}

evaluator2.setVariableValues(twoParamValues)
results2 = evaluator2.apply(evaluationName="twoParameterStudy", saveComparison=False)
```

With `saveComparison=False` only the numbers are kept, not the surfaces.

```bash
python parameterStudies.py
```

Both studies together take under a minute. The output goes to `customEvaluations/singleParameterSweep/` and `customEvaluations/twoParameterStudy/`.

Unlike the local study, the 3 x 3 grid includes the corners, where both parameters are changed together. Comparing the corner (0.8, 1.2) with the corner (1.2, 1.2) shows the interaction between the two rates directly.

## Part 3: Global Sensitivity

Script: `examples/sensitivityAnalysis/globalSensitivityStudy.py`

A global study samples the whole parameter range and computes Sobol indices: the share of the variance of the objective that each parameter is responsible for.

```python
gss = fit.GlobalSensitivityStudy("gss1", p1)

gss.loadProcessSequence(processPath)

# Declare all parameters used by the process sequence
gss.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Keep the remaining parameters fixed at their optimal values
gss.setFixedParameters(
    {name: optimum[name] for name in ["ionEnergy", "neutralStickP", "ionPowerCosine"]}
)

# Set variable parameters with ranges
gss.setVariableParameters(
    {
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)

# Same distance metric as used in the optimization
gss.setDistanceMetric("CCH")

# First-order and total-order indices only: numSamples * (numParams + 2) = 128
# evaluations. Second-order indices (secondOrder=True) need at least 500 samples
# per variable parameter, i.e. 1000 * (2 * 2 + 2) = 6000 evaluations here.
gss.setSamplingOptions(numSamples=32, secondOrder=False)

# Run the sensitivity analysis
gss.apply(saveComparison=False)
```

The number of evaluations is set by `setSamplingOptions`:

| Setting | Evaluations | Here |
|---|---|---|
| `secondOrder=False` | `numSamples * (numParams + 2)` | 32 * 4 = 128 |
| `secondOrder=True` | `numSamples * (2 * numParams + 2)` | needs far more samples, 6000 evaluations |

```bash
python globalSensitivityStudy.py
```

The 128 evaluations take about 5-10 minutes. The output goes to `projects/exampleProject/globSensStudies/gss1/`:

| File | Content |
|---|---|
| `parameter_samples.csv` | The sampled parameter sets |
| `objective_values.csv` | The objective value of each sample |
| `evaluation_results.json` | Detailed result of each evaluation |
| `sensitivity_results.json` | The Sobol indices |

### Reading the result

Two indices are reported per parameter, each with a confidence interval (`firstOrderConf`, `totalOrderConf`):

- **First-order index S1** (`firstOrder`): the share of the variance explained by this parameter alone.
- **Total-order index ST** (`totalOrder`): the share explained by this parameter including its interactions with the others.

How to read them:

- **ST close to zero**: the parameter does not matter over this range and can be fixed.
- **ST clearly larger than S1**: the parameter acts mainly through interactions. For the two rates here that is expected, because of the trade-off between them.
- **Indices that are noisy or slightly negative**: the sample is too small. 32 samples are enough for a ranking, not for precise values; increase `numSamples` for those.

!!! warning "Global is not local"
    A global study describes the whole range inside the bounds, where most parameter sets fit badly. It tells you which parameters decide between a bad and a reasonable fit. How well a parameter is pinned down near the optimum is a question for the local study.

## Key Takeaways

- All studies reuse the process sequence and the optimum stored in the optimization run.
- A local study is cheap and shows how well each parameter is determined at the optimum, one axis at a time.
- Sweeps with the `CustomEvaluator` are the quickest way to look at specific values or at two parameters together.
- A global study ranks parameters over their full range and exposes interactions, at a much higher cost.

## Next Steps

- [Tutorial 4: Multi-Domain Optimization](tutorial-4-multi-domain.md) - fit one parameter set to several geometries
- [Core Concepts](../getting-started/concepts.md) - background on the study types
