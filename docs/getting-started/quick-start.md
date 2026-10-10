# Quick Start

Run your first optimization with the example scripts that ship with ViennaFit: a deposition model is fitted to a trench profile.

**Estimated time**: 15-20 minutes, most of it waiting for the optimization

## Overview

In this quick start, you'll:

1. Create a ViennaFit project
2. Build the initial and target domains from two annotated contours
3. Run an optimization
4. Look at the results

The scripts are short and meant to be read. This page only shows their key lines; [Tutorial 1](../tutorials/tutorial-1-basic-optimization.md) explains them step by step.

## Prerequisites

- ViennaFit installed ([Installation Guide](installation.md))
- A copy of the ViennaFit repository, for the `examples/` folder
- Python 3.10+

## The Example

A trench in SiO2 is filled by a deposition step. Two contours describe the geometry before and after:

| File in `examples/0-example-data/` | Role |
|---|---|
| `regular-cropped-SiO2.dat` | Initial trench, 140 nm wide and 300 nm deep |
| `regular-cropped-Nitride.dat` | Target: the deposited film, with an overhang at the trench opening |

The optimization looks for the parameters of a neutral-plus-ion deposition model that turn the first contour into the second.

## Step 1: Create the Project

```bash
cd examples/1-setup
python initializeProject.py
```

The key line is

```python
project = fit.Project("exampleProject", projectPath).initialize()
```

which creates the folder `projects/exampleProject` in the repository root. The script then copies the two `.dat` files into `domains/annotations/` of the project.

## Step 2: Assign the Domains

```bash
python assignDomains.py
```

The script reads the two contours, converts them to level sets and stores them in the project:

```python
# Contour file -> polyline mesh
extentBottom = fit.readPointsFromFile(annotationBottom, meshBottom, gridDelta, mode="2D")
extentTarget = fit.readPointsFromFile(annotationTarget, meshTarget, gridDelta, mode="2D")

# Polyline mesh -> level set
vls.FromSurfaceMesh(domainBottom, meshBottom).apply()
vls.FromSurfaceMesh(domainTarget, meshTarget).apply()

# Initial geometry as a ViennaPS domain, target as a level set
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.SiO2)
p1.setInitialDomain(domainInitial)
p1.setTargetLevelSet(domainTarget)
```

!!! tip "Look at the domains"
    Open these two files in ParaView and overlay them before going on:

    ```
    projects/exampleProject/domains/initialDomain/exampleProject-initialDomain-surface.vtp
    projects/exampleProject/domains/targetDomain/exampleProject-targetDomain-surface.vtp
    ```

## Step 3: Run the Optimization

```bash
cd ../2-optimization
python basicOptimization.py
```

The script defines the process sequence, a function that runs the simulation for one set of parameters and returns the resulting surface:

```python
def processSequence1(domain: vps.Domain, params: dict[str, float]):
    model = vps.MultiParticleProcess()
    model.addNeutralParticle({vps.Material.SiO2: params["neutralStickP"]}, label="neutral")
    model.addIonParticle(
        sourcePower=params["ionPowerCosine"],
        meanEnergy=params["ionEnergy"],
        label="ion",
    )
    ...
    process.apply()
    return domain.getLevelSets()[-1]
```

and then tells ViennaFit which parameters to fit and how to score a result:

```python
opt1 = fit.Optimization(p1)
opt1.setProcessSequence(processSequence1)

opt1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)
opt1.setVariableParameters(
    {
        "neutralStickP": (0.001, 0.9),
        "ionPowerCosine": (1.0, 900.0),
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)
opt1.setFixedParameters({"ionEnergy": 100.0})

opt1.setDistanceMetrics(primaryMetric="CCH", additionalMetrics=["CA"])
opt1.setOptimizer("cma")
opt1.setName("run1")

opt1.apply(numEvaluations=200, saveComparison=True)
```

Four parameters are fitted within their bounds, one is fixed. The score is the Chamfer distance (`CCH`) between the simulated and the target surface.

The 200 evaluations take about 10-15 minutes on a desktop machine.

## Step 4: Look at the Results

The run is stored in `projects/exampleProject/optimizationRuns/run1/`:

```
run1/
├── run1-final-results.json   # Best score and best parameters
├── progressAll.csv           # Every evaluation
├── progressBest.csv          # Only the improvements
├── progress/                 # Surface of each new best evaluation (.vtp)
├── plots/                    # Convergence and parameter plots
├── run1-processSequence.py   # Copy of the process sequence
└── notes.txt
```

Start with these three:

- **`run1-final-results.json`** for the best parameters and the best score. A score of a few nm means the simulated surface lies within about one grid cell of the target.
- **`plots/run1-convergence-best.png`** to see whether the optimization has levelled off.
- **`domains/optimalDomains/run1-NNN.vtp`** in ParaView, overlaid with the target surface, to judge the fit by eye.

!!! note
    The simulation and the optimizer are both stochastic, so your numbers will differ slightly from run to run. Running the script again creates `run1_1`, `run1_2`, and so on; nothing is overwritten.

## Next Steps

1. **[Core Concepts](concepts.md)** - Projects, domains, metrics and parameters in more depth
2. **[Tutorial 1: Basic Optimization](../tutorials/tutorial-1-basic-optimization.md)** - The same workflow, explained line by line
3. **[Tutorial 2: Custom Evaluation](../tutorials/tutorial-2-custom-evaluation.md)** - Explore the neighbourhood of the optimum
4. **[Tutorial 3: Sensitivity Analysis](../tutorials/tutorial-3-sensitivity-analysis.md)** - Find out which parameters matter

## Common Next Questions

??? question "How do I use my own data?"
    Replace the two `.dat` files by your own contours: one point per line as `x y` (in nm), ordered along the surface. Then adapt the level set bounds and the material in `assignDomains.py`, and the process sequence in `basicOptimization.py`.

??? question "How do I choose the right distance metric?"
    Start with **CCH** (Chamfer distance). It measures how far the two surfaces are apart, in the length unit of your geometry, which makes the score easy to interpret.

    - Use **CA** when only the amount of material matters
    - Use **CCD** when specific dimensions matter (e.g. trench depth)
    - Use **CSF** for a comparison on the level set grid

    See [Core Concepts - Distance Metrics](concepts.md#distance-metrics) for details.

??? question "How many evaluations do I need?"
    It depends on the number of parameters and on how noisy the simulation is. The example uses 200 evaluations for 4 parameters.

    Check the convergence plot: if the best score is still dropping at the end, run more evaluations.

??? question "What if the optimization is interrupted?"
    An interrupted run has its progress files, but no final results file, best domain or plots. These can be written afterwards by finalizing the run:

    ```python
    project.finalizeOptimizationRun("run1")
    ```

    See [Tutorial 1](../tutorials/tutorial-1-basic-optimization.md#if-a-run-was-stopped-early) for details.

??? question "How do I use a different optimizer?"
    Change the optimizer with `setOptimizer()`:

    ```python
    opt1.setOptimizer("dlib")       # Default
    opt1.setOptimizer("cma")        # CMA-ES, used in the example
    opt1.setOptimizer("nevergrad")
    opt1.setOptimizer("ax")         # Bayesian optimization
    ```

    See [Core Concepts - Optimization](concepts.md#optimization) for details.
