# Core Concepts

This page explains the building blocks of ViennaFit. The code snippets use the deposition example from the `examples/` folder, the same one as in the [Quick Start](quick-start.md) and the tutorials.

## Projects

### What is a Project?

A **Project** is a container for all work related to one calibration task. It manages:

- Initial and target domains
- Optimization runs
- Custom evaluations
- Sensitivity studies

### Directory Structure

Initialization creates the `domains/` folder and `projectName-info.json`. The remaining folders are created on demand when you run the corresponding study:

```
projectName/
├── projectName-info.json         # Project metadata
├── domains/
│   ├── annotations/              # Contour data (.dat files)
│   ├── initialDomain/            # Starting geometry (.vpsd) and meshes
│   ├── targetDomain/             # Geometry to match (.lvst) and meshes
│   └── optimalDomains/           # Best result of each optimization run
├── optimizationRuns/             # Optimization results
│   └── run1/
│       ├── run1-final-results.json
│       ├── run1-processSequence.py
│       ├── progressBest.csv
│       ├── progressAll.csv
│       ├── progress/             # Surfaces of the improvements (.vtp)
│       └── plots/
├── customEvaluations/            # Custom parameter explorations
├── locSensStudies/               # Local sensitivity results
└── globSensStudies/              # Global sensitivity results
```

### Creating and Loading a Project

```python
import viennafit as fit

# Create a new project
project = fit.Project("exampleProject", "./projects").initialize()

# Or load an existing one
project = fit.Project()
project.load("./projects/exampleProject")
```

!!! note
    A project stores absolute paths. If you move its folder, assign the domains again.

### Single vs Multi-Domain

**Single-domain** (default): one initial geometry and one target.

```python
project.setInitialDomain(domain)
project.setTargetLevelSet(target)
```

**Multi-domain**: several named initial geometries, each with its own target. One parameter set is fitted to all of them.

```python
project.addInitialDomain("narrowTrench", domain1)
project.addTargetLevelSet("narrowTrench", target1)

project.addInitialDomain("wideTrench", domain2)
project.addTargetLevelSet("wideTrench", target2)
```

Initial and target domains are paired by name. See [Tutorial 4](../tutorials/tutorial-4-multi-domain.md).

## Domains

### Initial Domain

The **initial domain** is a ViennaPS domain: the geometry before the process, with its materials. The process sequence runs on a copy of it.

### Target Domain

The **target** is a ViennaLS level set: the surface the simulation should reproduce, typically traced from a cross-section image. It is only used for comparison.

### From Annotated Contours to Domains

Both domains of the example are built from annotation files, as in `examples/1-setup/assignDomains.py`.

**Step 1: The annotation file**

A text file with one point per line, ordered along the surface contour:

```
0.0  116.3
25.0  116.2
50.0  116.1
75.0  116.2
...
```

In 2D each line is `x y`, in 3D `x y z`. The example uses nm.

**Step 2: Read the points**

```python
import viennals as vls
import viennafit as fit

gridDelta = 3  # nm

meshTarget = vls.Mesh()
extentTarget = fit.readPointsFromFile(
    annotationTarget,   # path to the .dat file
    meshTarget,
    gridDelta,
    mode="2D",          # "2D" or "3D"
)
```

`readPointsFromFile` connects the points into a polyline mesh and returns the extent `[minX, maxX, minY, maxY]` of the contour, rounded to `gridDelta`.

Optional arguments adjust the coordinates while reading:

- **`scaleFactor`**: scale all coordinates, e.g. `scaleFactor=1000` to convert μm to nm
- **`shiftX`, `shiftY`, `shiftZ`**: translate the contour, e.g. to align it with the other domain
- **`reflectX=True`**: mirror the contour in x

**Step 3: Convert to a level set**

```python
domainTarget = vls.Domain(
    [10, 410, -450, 200],  # [minX, maxX, minY, maxY] in nm
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

vls.FromSurfaceMesh(domainTarget, meshTarget).apply()
```

The initial contour is converted the same way and then inserted into a ViennaPS domain as a material:

```python
domainInitial = vps.Domain(
    gridDelta=gridDelta,
    xExtent=extentBottom[1] - extentBottom[0],
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY,
)
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.SiO2)
```

!!! tip
    Use the same bounds and the same `gridDelta` for the initial and the target level set, and check both surface meshes in ParaView before optimizing.

Domains do not have to come from annotations. Any ViennaPS domain (for example from the ViennaPS geometry builders) can be the initial domain, and any ViennaLS level set can be the target.

## Process Sequences

### What is a Process Sequence?

A **process sequence** is a Python function that:

1. Takes a domain and a dictionary of parameters
2. Runs a ViennaPS simulation
3. Returns the resulting level set, which is compared with the target

### Function Signature

**Single-domain:**
```python
def processSequence(
    domain: vps.Domain,         # Copy of the initial domain
    params: dict[str, float]    # Parameter values for this evaluation
) -> vls.Domain:                # Resulting level set
    ...
```

**Multi-domain:**
```python
def processSequence(
    domains: dict[str, vps.Domain],     # Named copies of the initial domains
    params: dict[str, float]
) -> dict[str, vls.Domain]:             # Named results
    ...
```

ViennaFit passes a fresh copy of the initial domain for every evaluation, so the function can modify it freely.

### Example

The process sequence of `examples/2-optimization/basicOptimization.py`:

```python
def processSequence1(domain: vps.Domain, params: dict[str, float]):
    model = vps.MultiParticleProcess()

    # Set the parameters for the neutral
    sticking = {vps.Material.SiO2: params["neutralStickP"]}
    model.addNeutralParticle(sticking, label="neutral")

    # Set the parameters for the ion
    model.addIonParticle(
        sourcePower=params["ionPowerCosine"],
        meanEnergy=params["ionEnergy"],
        label="ion",
    )

    def rateFunction(fluxes, material):
        if material == vps.Material.SiO2:
            return (
                fluxes[0] * params["neutralRate"] + fluxes[1] * params["ionRate"]
            )
        return 0.0

    model.setRateFunction(rateFunction)

    process = vps.Process()
    process.setFluxEngineType(vps.FluxEngineType.CPU_DISK)
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(1.0)

    rayTracing = vps.RayTracingParameters()
    rayTracing.raysPerPoint = 300
    process.setParameters(rayTracing)

    process.apply()

    return domain.getLevelSets()[-1]
```

The positive rate makes the surface grow, so this is a deposition. `domain.getLevelSets()[-1]` is the topmost level set, the surface after the process.

A sequence may contain as many process steps as needed, as long as it returns one level set per domain.

### Where the Sequence is Stored

Pass the function with `setProcessSequence(function)`, or keep it in its own file and use `loadProcessSequence("path/to/sequence.py")`. Either way a copy is written into the run folder (`run1-processSequence.py`), so later studies can reuse exactly the model that was fitted.

## Distance Metrics

### What are Distance Metrics?

A **distance metric** reduces the difference between the simulated and the target geometry to a single number. Lower is better, and the optimizer minimizes it.

### Available Metrics

| Metric | Name | What it measures |
|--------|------|------------------|
| **CCH** | Chamfer | RMS distance between the two surfaces, in the length unit of the geometry |
| **CA** | Area | Area in which the two geometries do not overlap |
| **CSF** | Sparse field | RMS difference of the level set values on the grid points at the surface |
| **CCD** | Critical dimensions | RMS error of selected dimensions |

Further variants (`CSF-IS`, `CNB`, `CA+CSF`, `CA+CNB`) are available as well.

### Choosing a Metric

- **CCH** is a good default. Its value is a distance (nm in the example), so "CCH = 3" directly says the surfaces are about 3 nm apart.
- **CA** only sees how much the geometries differ in area. Use it when the amount of material matters more than the shape.
- **CSF** compares on the level set grid and reacts to small local differences.
- **CCD** ignores everything except the dimensions you select.

### Setting the Metric

```python
opt1.setDistanceMetrics(
    primaryMetric="CCH",       # Minimized by the optimizer
    additionalMetrics=["CA"],  # Recorded for every evaluation, not optimized
)
```

Additional metrics appear as extra columns in the progress files (`CA_value`, ...). They are useful as a cross-check: two metrics that disagree about which result is better tell you something about the fit.

Studies other than the optimization use `setDistanceMetric("CCH")`, and the `CustomEvaluator` adds further metrics with `setAdditionalMetrics(["CSF"])`.

### CCD - Critical Dimensions

CCD needs to know which dimensions to measure. Each range scans along one axis and takes the minimum or maximum of the surface there:

```python
opt1.setDistanceMetrics(
    primaryMetric="CCD",
    criticalDimensionRanges=[
        {
            "axis": "x",          # Scan along x ...
            "min": 180,           # ... between x = 180
            "max": 240,           # ... and x = 240
            "findMaximum": False  # and take the lowest point: the trench bottom
        },
    ],
)
```

### CSF - Expansion Width

For CSF the target level set is expanded before the comparison so that it overlaps the simulated surface. The width can be set with `sparseFieldExpansionWidth` (default 200) in `setDistanceMetrics`.

## Parameters

Every key that the process sequence reads from `params` is a parameter. Each one is declared and then made either variable or fixed.

```python
# Declare all parameters used by the process sequence
opt1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Variable parameters: fitted within (lowerBound, upperBound)
opt1.setVariableParameters(
    {
        "neutralStickP": (0.001, 0.9),
        "ionPowerCosine": (1.0, 900.0),
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)

# Fixed parameters: constant
opt1.setFixedParameters({"ionEnergy": 100.0})
```

**Rules:**

1. Call `setParameterNames` first.
2. The names must match the keys used in the process sequence.
3. Every declared parameter is either variable or fixed, never both.

### Choosing Bounds

- Cover every value you consider plausible, but not much more: very wide bounds slow the optimization down.
- Know what a unit of each parameter does. In the example one unit of `ionRate` deposits about ten times as much as one unit of `neutralRate`, which is why their ranges differ so much.
- If the optimum sits on a bound, widen that bound and run again.
- For parameters whose bounds span several orders of magnitude, `setLogScaledParameters([...])` lets the `cma` optimizer search them on a logarithmic scale.

### When to Fix a Parameter

- Its value is known from the experiment
- The target does not constrain it (see [Sensitivity Analysis](#sensitivity-analysis))
- You want fewer dimensions for a first, faster run

## Optimization

### How Optimization Works

1. **Propose**: the optimizer chooses a set of parameter values within the bounds
2. **Evaluate**: the process sequence runs on a copy of the initial domain, and the result is compared with the target
3. **Repeat**: until the requested number of evaluations is reached

```python
opt1 = fit.Optimization(project)
opt1.setProcessSequence(processSequence1)
# ... parameters and metrics as above ...
opt1.setOptimizer("cma")
opt1.setName("run1")
opt1.setNotes("Basic optimization of the multiparticle process for deposition.")

opt1.apply(numEvaluations=200, saveComparison=True)
```

### Optimizers

| Name | Method | Notes |
|------|--------|-------|
| `"dlib"` | Global optimization | Default |
| `"cma"` | CMA-ES (evolution strategy) | Used in the example; copes well with noisy objectives |
| `"nevergrad"` | Nevergrad | Adapts its strategy during the run |
| `"ax"` / `"botorch"` | Bayesian optimization | Works in batches; configured with `setNumBatches()` and `setBatchSize()` instead of `numEvaluations` |

Related settings:

- **`setStartingPoint({...})`**: start from given values instead of the centre of the bounds
- **`setEarlyStopping(patienceEvaluations=...)`**: stop when there has been no improvement for that many evaluations

### Results

A run called `run1` writes to `optimizationRuns/run1/`:

| File | Content |
|---|---|
| `run1-final-results.json` | Best score, best parameters, bounds, optimizer, number of evaluations |
| `progressAll.csv` | Every evaluation: parameters, metric values, timings |
| `progressBest.csv` | Only the evaluations that improved on the best so far |
| `progress/` | Surface of each new best evaluation |
| `plots/` | Convergence and parameter plots |
| `run1-processSequence.py` | Copy of the process sequence |
| `notes.txt` | Text given to `setNotes` |

The best surface is also copied to `domains/optimalDomains/`.

```json
{
    "bestScore": 2.65,
    "bestEvaluation#": 202,
    "bestParameters": {
        "ionEnergy": 100.0,
        "neutralStickP": 0.408,
        "ionPowerCosine": 92.0,
        "neutralRate": 96.2,
        "ionRate": 2.08
    }
}
```

If a run with the same name exists, the new one is stored as `run1_1`, `run1_2`, and so on. A run that was interrupted can be completed with `project.finalizeOptimizationRun("run1")`.

## Custom Evaluation

A `CustomEvaluator` runs the process sequence of an existing optimization run on parameter values that you choose. Parameters you do not mention stay at the optimum of that run.

```python
evaluator = fit.CustomEvaluator(project)
evaluator.loadOptimizationRun("run1")
evaluator.setDistanceMetric("CCH")
```

### Grid

All combinations of the listed values:

```python
evaluator.setVariableValues({
    "neutralRate": [70.0, 85.0, 100.0, 115.0, 130.0],
    "ionRate": [1.0, 2.0, 3.0, 4.0],
})

results = evaluator.apply(evaluationName="parameterSweep1", saveComparison=True)
evaluator.saveGridReport()
```

This gives 5 x 4 = 20 evaluations.

### Specific Combinations

Only the parameter sets you list:

```python
evaluator.setVariableValuesPaired([
    {"neutralRate": 85.0, "ionRate": 3.0},
    {"neutralRate": 100.0, "ionRate": 2.0},
    {"neutralRate": 115.0, "ionRate": 1.0},
])

results = evaluator.apply(evaluationName="specificCombinations")
```

### Repeatability

The same parameters several times, to measure the noise of the simulation:

```python
bestParams = evaluator.getOptimalParameters()
evaluator.setConstantParametersWithRepeats(bestParams, numRepeats=10)

results = evaluator.apply(evaluationName="repeatedEvaluation")
```

Differences between two results that are smaller than this noise are not meaningful.

See [Tutorial 2](../tutorials/tutorial-2-custom-evaluation.md).

## Sensitivity Analysis

**Sensitivity analysis** tells you which parameters the fit actually depends on:

- Which parameters are well determined by the target?
- Which can be fixed without making the fit worse?
- Do parameters compensate for each other?

### Local Sensitivity

One parameter at a time is moved through a range around a point of interest, usually the optimum.

```python
lss1 = fit.LocalSensitivityStudy("lss1", project)
lss1.loadProcessSequence(processPath)
lss1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)
lss1.setFixedParameters(
    {name: optimum[name] for name in ["ionEnergy", "neutralStickP", "ionPowerCosine"]}
)

# (lowerBound, pointOfInterest, upperBound)
lss1.setParameterSensitivityRanges(
    {
        "neutralRate": (1.0, optimum["neutralRate"], 150.0),
        "ionRate": (0.1, optimum["ionRate"], 10.0),
    }
)

lss1.setDistanceMetric("CCH")
lss1.apply()
```

**Use when** you have an optimum and want to know how sharply each parameter is determined. It is cheap, but it does not show interactions between parameters.

### Global Sensitivity

The whole parameter range is sampled and Sobol indices are computed.

```python
gss = fit.GlobalSensitivityStudy("gss1", project)
# ... process sequence, parameter names and fixed parameters as above ...

gss.setVariableParameters(
    {
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)

gss.setDistanceMetric("CCH")
gss.setSamplingOptions(numSamples=32, secondOrder=False)
gss.apply(saveComparison=False)
```

**Sobol indices:**

- **S1** (first-order): effect of the parameter on its own
- **ST** (total-order): effect including interactions with other parameters
- **S2** (second-order, with `secondOrder=True`): pairwise interactions; needs far more samples

**Use when** you want a ranking of the parameters over their full range and can afford the evaluations: `numSamples * (numParams + 2)` without second-order indices.

See [Tutorial 3](../tutorials/tutorial-3-sensitivity-analysis.md).

## Summary

**Projects** hold everything → **Domains** define the problem → **Process sequences** run the simulation → **Distance metrics** score the result → **Optimizers** search for the best parameters

Key takeaways:

- Set the dimension in ViennaPS and ViennaLS before creating domains
- Check the initial and target surface before optimizing
- Declare every parameter, with bounds or a fixed value
- After a run, check the fit, the convergence and whether a parameter sits on a bound
- Use custom evaluations and sensitivity studies to learn how far the result can be trusted

## Next Steps

- **[Tutorial 1](../tutorials/tutorial-1-basic-optimization.md)**: The optimization workflow, step by step
- **[Tutorial 2](../tutorials/tutorial-2-custom-evaluation.md)**: Custom parameter exploration
- **[Tutorial 3](../tutorials/tutorial-3-sensitivity-analysis.md)**: Sensitivity analysis in practice
