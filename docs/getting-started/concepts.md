# Core Concepts

## Projects

### What is a Project?

A **Project** is a container for all work related to one calibration or optimization task. It manages:

- Initial and target domains
- Optimization runs
- Sensitivity studies
- Custom evaluations

### Directory Structure

When you initialize a project, ViennaFit creates:

```
projectName/
├── projectName-info.json         # Project metadata
├── domains/
│   ├── initialDomain/            # Starting geometries (.vpsd files)
│   ├── targetDomain/             # Goals to match (.lvst files)
│   ├── annotations/              # Measurement data
│   └── optimalDomains/           # Best results
├── optimizationRuns/             # Optimization results
│   └── run1/
│       ├── run1-final-results.json
│       ├── progressBest.csv
│       ├── progressAll.csv
│       └── progress/             # Visualization files
├── customEvaluations/            # Custom parameter explorations
├── localSensitivityStudies/      # Local sensitivity results
└── globalSensitivityStudies/     # Global sensitivity results
```

### Creating a Project

```python
import viennafit as fit

# Create new project
project = fit.Project(
    name="etchCalibration",
    projectPath="./projects"
).initialize()

# Or load existing project
project = fit.Project()
project.load("./projects/etchCalibration")
```

### Single vs Multi-Domain

**Single-Domain** (default):
- One initial geometry
- One target to match
- Simplest case

**Multi-Domain**:
- Multiple initial geometries with names
- Multiple targets to match

```python
# Single-domain (backward compatible)
project.setInitialDomain(domain)
project.setTargetLevelSet(target)

# Multi-domain (new in 2.0)
project.addInitialDomain("wafer1", domain1)
project.addInitialDomain("wafer2", domain2)
project.addTargetLevelSet("wafer1", target1)
project.addTargetLevelSet("wafer2", target2)
```

!!! info "Domain Naming"
    Use descriptive names like `"frontSide"`, `"backSide"`, not just `"domain1"`.

    The name `"default"` is reserved for single-domain compatibility.

## Domains

### Initial Domain

The **initial domain** is a ViennaPS domain. It is your starting geometry before any processing.

**Contains:**
- Material layers (Si, SiO2, etc.)
- Initial geometry
- Spatial extent and grid resolution

**Created with ViennaPS:**
```python
import viennaps as vps

initial = vps.Domain(
    gridDelta=5.0,              # Grid spacing (nm)
    xExtent=400.0,              # Width (nm)
    yExtent=200.0,              # Height (nm)
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY
)

# Add material layers
initial.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)

# See geometry creation tools in ViennaPS:
# psMakeFin, psMakeHole, psMakeTrench, psMakePlane, psMakeStack...
```

### Target Domain

The **target domain** is what you want your simulation to match - typically from experimental data.

**Source:**
- Experimental measurements (SEM cross-sections etc.)
- Loaded from annotated point data (.dat files) or VTK files (.vtp, .vtu)
- Converted to level set representation

#### Creating Target from Annotated Experimental Data

The most common workflow uses the `readPointsFromFile` utility to convert experimental measurements into target domains:

**Step 1: Prepare Annotation Data**

Annotate your experimental images (SEM, TEM, etc.) by tracing material boundaries and extracting coordinate points. Save as a `.dat` / `.txt` / `.csv` file with space-separated coordinates:

**2D format** (x y per line):
```
-100.0 0.0
-95.0 0.0
-90.0 -2.5
-85.0 -7.5
...
```

**3D format** (x y z per line):
```
50.0 100.0 0.0
52.0 98.5 0.0
54.0 97.0 0.0
...
```

Points should be **sequential** along the surface contour. Units are typically nanometers (nm) but can be set in ViennaPS accordingly.

**Step 2: Load Points and Create Target Domain**

```python
import viennals as vls
import viennafit as fit

# Create mesh to hold the points
meshTarget = vls.Mesh()

# Read points from file and get extent
gridDelta = 5.0  # Grid resolution (nm)
extentTarget = fit.readPointsFromFile(
    "experimental_profile.dat",
    meshTarget,
    gridDelta,
    mode="2D",           # "2D" or "3D"
    reflectX=False,      # Mirror in X axis if needed
    shiftX=0.0,          # Translate X coordinates if needed
    shiftY=0.0,          # Translate Y coordinates if needed
    scaleFactor=1.0      # Scale all coordinates if needed
)

# Create level set domain with computed extent
target = vls.Domain(
    extentTarget,
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

# Convert mesh to level set representation
vls.FromSurfaceMesh(target, meshTarget).apply()
```

**Common transformations:**
- **`reflectX=True`**: Mirror coordinates if annotation coordinate system is flipped
- **`shiftX/Y/Z`**: Center or align geometry to match initial domain
- **`scaleFactor`**: Convert units (e.g., `scaleFactor=1000` for μm → nm)

**Example:** See `examples/setup/loadExperimentalData.py` for a complete demonstration using `examples/data/u_shaped_trench.dat`.

#### Creating Target Programmatically

For testing or synthetic cases, create targets directly:

```python
import viennals as vls

target = vls.Domain([xmin, xmax, ymin, ymax], boundary, gridDelta)

# Create geometric shape
vls.MakeGeometry(target, vls.lsBox([x, y], [width, height])).apply()
```

### Domain Pairing

In multi-domain mode, initial and target domains are paired by name:

```python
# Initial domain "wafer1" pairs with target "wafer1"
project.addInitialDomain("wafer1", initial1)
project.addTargetLevelSet("wafer1", target1)

# Different geometry, same process parameters
project.addInitialDomain("wafer2", initial2)
project.addTargetLevelSet("wafer2", target2)
```

Optimization finds parameters that work well across all pairs.

## Process Sequences

### What is a Process Sequence?

A **process sequence** is a Python function that:

1. Takes a domain and parameters as input
2. Runs a ViennaPS simulation
3. Returns the resulting level set

### Function Signature

**Single-Domain:**
```python
def processSequence(
    domain: vps.Domain,         # Initial geometry (copied)
    params: dict[str, float]    # Parameters chosen by optimizer
) -> vls.Domain:                # Resulting level set
    # Your simulation code
    return result
```

**Multi-Domain:**
```python
def processSequence(
    domains: dict[str, vps.Domain],     # Named domains
    params: dict[str, float]            # Parameters
) -> dict[str, vls.Domain]:             # Named results
    results = {}
    for name, domain in domains.items():
        # Process each domain
        results[name] = processOne(domain, params)
    return results
```

!!! warning "Domain Copying"
    ViennaFit copies domains before passing to your function. Modify the copy freely - the original is preserved.

### Example Process Sequence

```python
def etchProcess(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    """
    Simple plasma etching process.

    Parameters:
    - ionFlux: Ion flux density
    - neutralFlux: Neutral flux density
    - meanEnergy: Ion energy
    """
    # Create process model
    model = vps.MultiParticleProcess()

    # Configure with parameters
    model.addIonParticle(
        sourcePower=params["ionFlux"],
        meanEnergy=params["meanEnergy"],
        label="ion"
    )

    model.addNeutralParticle(
        sticking={vps.Material.Si: params["neutralSticking"]},
        label="neutral"
    )

    # Define rate function
    def rateFunction(fluxes, material):
        if material == vps.Material.Si:
            return (fluxes["ion"] * params["ionEtchRate"] +
                    fluxes["neutral"] * params["neutralEtchRate"])
        return 0.0

    model.setRateFunction(rateFunction)

    # Run simulation
    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(1.0)  # 1 second
    process.apply()

    # Return result
    return domain.getLevelSets()[-1]
```

### Multi-Step Processes

Process sequences can include multiple steps:

```python
def complexProcess(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    # Step 1: Deposition
    deposit = vps.IsotropicProcess(rate=params["depositRate"])
    vps.Process(domain, deposit, params["depositTime"]).apply()

    # Step 2: Etch
    etch = vps.IsotropicProcess(rate=-params["etchRate"])
    vps.Process(domain, etch, params["etchTime"]).apply()

    # Step 3: Planarization
    vps.Planarize(domain, 0.0).apply()

    return domain.getLevelSets()[-1]
```

## Distance Metrics

### What are Distance Metrics?

**Distance metrics** quantify how well your simulation matches the target. They return a single number (lower = better match).

### Available Metrics

| Metric | Full Name | Best For | Speed | Sensitivity |
|--------|-----------|----------|-------|-------------|
| **CA** | Compare Area | Area matching | Fast | Coarse |
| **CCH** | Compare Chamfer | Shape similarity | Fast | High |
| **CSF** | Compare Sparse Field | Detailed matching | Fast | Very High |
| **CCD** | Compare Critical Dimensions | Specific measurements | Fast | Targeted |

### Choosing a Metric

**Decision Flow:**
```
What matters most?

├─ Specific dimensions (depth, width)?
│  └─ Use CCD with measurement points

├─ Overall shape similarity?
│  └─ Use CCH (Chamfer distance)

├─ Simple area difference?
│  └─ Use CA (fastest)

└─ Very detailed matching?
   └─ Use CSF (slowest, most sensitive)
```

### CA - Compare Area

Measures difference in total area/volume.

**Formula:**
```
CA = |Area(simulation) - Area(target)|
```

**Use when:**
- You care about total material removed/added
- Shape details don't matter

**Example:**
```python
opt.setDistanceMetrics(primaryMetric="CA")
```

### CCH - Compare Chamfer

Chamfer distance: average distance from each surface point to nearest target point.

**Bidirectional:**
- Forward: simulation → target
- Backward: target → simulation
- Final = max(forward, backward)

**Use when:**
- Shape similarity is important
- You want robust matching
- Good default choice

**Example:**
```python
opt.setDistanceMetrics(primaryMetric="CCH")
```

### CSF - Compare Sparse Field

Sum of squared differences at grid points in narrow band.

**Very sensitive** to small differences.

**Use when:**
- You need very precise matching
- Computation time is acceptable
- Small shape variations matter

**Example:**
```python
opt.setDistanceMetrics(
    primaryMetric="CSF",
    sparseFieldExpansionWidth=200  # This is explained in the function itself
)
```

### CCD - Compare Critical Dimensions

RMSE of specific dimension measurements.

**Requires measurement points:**
```python
opt.setDistanceMetrics(
    primaryMetric="CCD",
    criticalDimensionRanges=[
        {
            'axis': 'y',           # Measure in y direction
            'min': 80,             # x position range
            'max': 120,
            'findMaximum': True    # Find max y value
        },
        {
            'axis': 'x',
            'min': -50,            # y position range
            'max': -30,
            'findMaximum': False   # Find min x value
        }
    ]
)
```

**Use when:**
- Specific dimensions are critical (trench depth, width, etc.)
- You have measurement specifications
- Other shape variations are acceptable

### Combined Metrics

Use multiple metrics:

```python
opt.setDistanceMetrics(
    primaryMetric="CSF",           # Optimize using combined
    additionalMetrics=["CCH", "CCD"]  # Track but don't optimize
)
```

**Primary** metric is optimized. **Additional** metrics are computed and tracked for analysis.

## Parameters

### Variable Parameters

**Variable parameters** are what you're optimizing. Each needs bounds.

```python
opt.setVariableParameters({
    "etchRate": (10.0, 100.0),      # Min 10, Max 100
    "stickingProb": (0.01, 0.99),   # Min 0.01, Max 0.99
    "ionEnergy": (5.0, 150.0)       # Min 5, Max 150
})
```

**Bounds should:**
- Cover physically realistic range
- Not be too wide (slows convergence)
- Not be too narrow (misses optimum)

**Tips:**
- Start with wide bounds, narrow after initial run
- Use log scale for parameters spanning orders of magnitude
- Check physics, set a range which makes sense

### Fixed Parameters

**Fixed parameters** stay constant during optimization.

```python
opt.setFixedParameters({
    "temperature": 300.0,    # Room temperature
    "pressure": 1.0,         # 1 Torr
    "gridDelta": 5.0         # Grid resolution
})
```

**Use fixed parameters for:**
- Known values from experimental conditions
- Parameters determined separately
- Simplifying optimization (fewer dimensions)

### Parameter Names

Parameter names must be declared **before** setting fixed and variable parameters using `setParameterNames()`.

**Important rules:**

1. **All parameters must be declared**: Every parameter used in your process sequence must appear in the parameter names list
2. **No overlap**: A parameter cannot be both fixed and variable
3. **Complete partition**: Every declared parameter must be classified as either fixed or variable

**Example:**
```python
# Step 1: Declare all parameter names
opt.setParameterNames(["etchRate", "time", "temperature", "pressure"])

# Step 2: Classify parameters
opt.setVariableParameters({
    "etchRate": (10, 100),    # Will be optimized
    "time": (0.5, 5.0)        # Will be optimized
})

opt.setFixedParameters({
    "temperature": 300.0,     # Constant
    "pressure": 1.0           # Constant
})

# Variable + Fixed = All parameter names
# {"etchRate", "time"} ∪ {"temperature", "pressure"} = {"etchRate", "time", "temperature", "pressure"} ✓
```

**Matching process sequence:**

Parameter names must match the keys used in your process sequence:

```python
# In optimization setup
opt.setParameterNames(["etchRate", "time"])
opt.setVariableParameters({
    "etchRate": (10, 100),
    "time": (0.5, 5.0)
})

# In process sequence - use same key names
def process(domain, params):
    rate = params["etchRate"]  # Must match "etchRate"
    time = params["time"]      # Must match "time"
    # ...
```

## Optimization

### How Optimization Works

1. **Initialize**: Choose starting parameters (random or specified)
2. **Evaluate**: Run process sequence, compute distance metric
3. **Update**: Algorithm chooses next parameters to try
4. **Repeat**: Until convergence or max evaluations reached

### Optimization Algorithms

**dlib** (default):
- Global optimization
- Good for most cases
- Robust to noise

**Nevergrad:**
- Automatic adaptation of strategy durin optimization run
- Can be slower to converge but better at exploring local minima

**Ax/BoTorch:**
- Bayesian optimization
- Uses batches instead of single evaluations
- As a result usually needs less evaluations overall
- User chooses batch size etc.

```python
opt.setOptimizer("dlib")       # Default
opt.setOptimizer("nevergrad")  # Alternative
opt.setOptimizer("ax")         # Bayesian (requires extra packages)
```

### Convergence

Optimization stops when:

- Maximum evaluations reached
- Algorithm declares convergence
- You manually stop it

**Monitor convergence plots** to check if optimization has truly converged or just hit evaluation limit.

### Results

After optimization completes:

**Files created:**
```
optimizationRuns/runName/
├── runName-final-results.json    # Best parameters
├── progressBest.csv               # Improvement history
├── progressAll.csv                # All evaluations
├── plots/                         # Convergence plots
└── progress/                      # Visualization files
```

**Best parameters in JSON:**
```json
{
  "bestParameters": {
    "etchRate": 45.3,
    "stickingProb": 0.23,
    "ionEnergy": 75.8
  },
  "bestScore": 12.45,
  "bestEvaluation#": 87
}
```

## Sensitivity Analysis

### Purpose

**Sensitivity analysis** identifies which parameters matter most.

**Questions it answers:**
- Which parameters significantly affect results?
- Which can be fixed without hurting performance?
- How do parameters interact?

### Local Sensitivity

**One-at-a-time** variation around a point.

```python
study = fit.LocalSensitivityStudy("local1", project)
study.setParameterSensitivityRanges({
    "param1": (low, mid, high),    # Evaluate at 3 points
    "param2": (low, mid, high)
})
study.apply()
```

**Use when:**
- You have an optimum and want to understand nearby behavior
- Fast results needed
- Parameter interactions not critical

### Global Sensitivity

**Sobol indices** over entire parameter space.

```python
study = fit.GlobalSensitivityStudy("global1", project)
study.setVariableParameters({
    "param1": (min, max),
    "param2": (min, max)
})
study.setSamplingOptions(numSamples=100, secondOrder=True)
study.apply()
```

**Sobol Indices:**
- **S1** (first-order): Direct effect of parameter
- **ST** (total): Direct + interaction effects
- **S2** (second-order): Pairwise interactions

**Use when:**
- You want to understand global behavior
- Parameter interactions important
- You have computational budget

## Custom Evaluation

Beyond optimization, you can:

### Grid Search

Systematically evaluate parameter combinations:

```python
evaluator = fit.CustomEvaluator(project)
evaluator.loadOptimizationRun("run1")

evaluator.setVariableValues({
    "param1": [10, 20, 30, 40, 50],
    "param2": [0.1, 0.2, 0.3, 0.4]
})

results = evaluator.evaluateGrid("gridSearch1")
```

Creates 5 × 4 = 20 evaluations.

### Specific Combinations

Test exact parameter sets:

```python
evaluator.setVariableValuesPaired([
    {"param1": 25, "param2": 0.15},
    {"param1": 35, "param2": 0.25},
    {"param1": 45, "param2": 0.35}
])

results = evaluator.evaluateGrid("specific3")
```

### Repeatability Testing

Run identical parameters multiple times to assess variance:

```python
best = evaluator.loadBestFromProgressCSV("run1")

evaluator.setConstantParametersWithRepeats(best, numRepeats=10)
results = evaluator.evaluateGrid("repeatability")

# Analyze variance
import numpy as np
values = [r["objectiveValue"] for r in results]
print(f"Mean: {np.mean(values):.4f}")
print(f"Std: {np.std(values):.4f}")
```

## Summary

**Projects** organize everything → **Domains** define problem → **Process Sequences** run simulations → **Distance Metrics** score results → **Optimizers** find best parameters

Key takeaways:

- Always set dimensions before creating domains
- Choose distance metric based on what you're matching
- Start with reasonable parameter bounds
- Monitor convergence - more evaluations if still improving
- Use sensitivity analysis to understand your model

## Next Steps

- **[Tutorial 1](../tutorials/tutorial-1-basic-optimization.md)**: Complete optimization walkthrough
- **[Tutorial 2](../tutorials/tutorial-2-custom-evaluation.md)**: Custom parameter exploration
- **[Tutorial 3](../tutorials/tutorial-3-sensitivity-analysis.md)**: Sensitivity analysis in practice
