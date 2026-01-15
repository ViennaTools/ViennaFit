# Quick Start

Get your first optimization running in 15 minutes! This guide walks you through a complete optimization workflow from start to finish.

## Overview

In this quick start, you'll:

1. Create a ViennaFit project
2. Load initial and target domains
3. Define a simple process sequence
4. Configure and run an optimization
5. View and interpret results

**Estimated time**: 15 minutes

## Prerequisites

- ViennaFit installed ([Installation Guide](installation.md))
- Basic familiarity with ViennaPS
- Python 3.10+

## Step 1: Setup (2 minutes)

Create a new Python file (`my_first_optimization.py`) and import required modules:

```python
import viennafit as fit
import viennaps as vps
import viennals as vls

# Set dimensions (must be done before creating domains)
vps.setDimension(2)
vls.setDimension(2)
```

!!! tip "Dimension Setting"
    Always set dimensions at the start of your script. ViennaPS and ViennaLS require this before creating any domains or level sets.

## Step 2: Create Project (1 minute)

Initialize a new ViennaFit project:

```python
# Create and initialize project
project = fit.Project("myFirstOptimization", "./projects")
project.initialize()

print(f"Project created at: {project.projectPath}")
```

**What happened:**
- Created directory structure at `./projects/myFirstOptimization/`
- Generated project metadata file
- Created subdirectories for domains, results, and studies

## Step 3: Load Domains (3 minutes)

For this example, we'll create simple test domains. In a real scenario, you'd load experimental data.

### Create Initial Domain

```python
# Create initial domain (simple substrate)
gridDelta = 5.0
initialDomain = vps.Domain(
    gridDelta=gridDelta,
    xExtent=400.0,
    yExtent=200.0,
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY
)

# Add substrate layer
substrate = vls.Domain(
    [-200, 200, -100, 100],  # bounds: [xmin, xmax, ymin, ymax]
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(substrate, vls.lsBox([0, -50], [400, 30])).apply()

initialDomain.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)

# Assign to project
project.setInitialDomain(initialDomain)
print("Initial domain set")
```

### Create Target Domain

```python
# Create target domain (what we want to match)
targetLevelSet = vls.Domain(
    [-200, 200, -100, 100],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)

# Create target profile (e.g., etched trench)
vls.MakeGeometry(
    targetLevelSet,
    vls.lsBox([80, -50], [320, -20])  # Etched region
).apply()

# Assign to project
project.setTargetLevelSet(targetLevelSet)
print("Target domain set")
```

!!! info "Real-World Usage"
    In practice, you'd load target domains from experimental data:
    ```python
    targetLevelSet = vls.Domain(...)
    reader = vls.VTKReader(targetLevelSet)
    reader.apply("experimental_profile.vtp")
    project.setTargetLevelSet(targetLevelSet)
    ```

## Step 4: Define Process Sequence (5 minutes)

The process sequence is a function that simulates your process with given parameters:

```python
def processSequence(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    """
    Simple isotropic etching process.

    Parameters to optimize:
    - etchRate: Etching velocity (nm/s)
    - processTime: Etch duration (seconds)
    """
    # Create isotropic etch model
    model = vps.IsotropicProcess(rate=-params["etchRate"])

    # Set up process
    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(params["processTime"])

    # Run simulation
    process.apply()

    # Return resulting level set
    return domain.getLevelSets()[-1]
```

!!! tip "Process Sequence Tips"
    - Must accept `domain` and `params` arguments
    - Must return a `vls.Domain` (level set)
    - Can be as complex as needed (multi-step processes, etc.)
    - Params dict contains values chosen by optimizer

## Step 5: Configure Optimization (2 minutes)

Set up the optimization with parameter bounds and distance metric:

```python
# Create optimization object
opt = fit.Optimization(project)
opt.setProcessSequence(processSequence)

# Define parameters to optimize
opt.setParameterNames(["etchRate", "processTime"])

# Set parameter bounds (min, max)
opt.setVariableParameters({
    "etchRate": (5.0, 50.0),      # nm/s
    "processTime": (0.5, 3.0)     # seconds
})

# Choose distance metric
opt.setDistanceMetrics(primaryMetric="CA")  # Compare Area

# Name this optimization run
opt.setName("run1")
opt.setNotes("First optimization - testing CA metric")

print("Optimization configured")
```

!!! info "Distance Metrics"
    - **CA** (Compare Area): Simple area difference, fast
    - **CCH** (Chamfer): Shape similarity, recommended for most cases
    - **CSF** (Sparse Field): Detailed comparison, slower
    - **CCD**: Specific dimension matching

    See [Core Concepts](concepts.md#distance-metrics) for details.

## Step 6: Run Optimization (1 minute)

Execute the optimization:

```python
print("Starting optimization...")
opt.apply(
    numEvaluations=50,        # Try 50 parameter combinations
    saveVisualization=True     # Save domain visualizations
)
print("Optimization complete!")
```

**What to expect:**
- Progress printed to console
- Runtime: 2-10 minutes (depending on simulation complexity)
- Results saved automatically

**Console output example:**
```
Evaluation 1/50: Objective = 245.67
Evaluation 2/50: Objective = 189.23
Evaluation 3/50: Objective = 156.41
...
Best objective: 23.45
Best parameters: {'etchRate': 18.3, 'processTime': 1.47}
```

## Step 7: View Results (1 minute)

### Check Files Created

```python
import os

results_dir = os.path.join(project.projectPath, "optimizationRuns", "run1")
print(f"\nResults saved in: {results_dir}")
print("\nFiles created:")
for file in os.listdir(results_dir):
    print(f"  - {file}")
```

Expected files:
```
run1/
├── run1-final-results.json          # Best parameters found
├── run1-startingConfiguration.json  # Initial setup
├── run1-processSequence.py          # Copy of process function
├── notes.txt                        # Your notes
├── progressBest.csv                 # History of improvements
├── progressAll.csv                  # All evaluations
├── progress/                        # Visualization files (.vtp)
└── plots/                           # Convergence plots
```

### View Best Parameters

```python
import json

# Load results
with open(os.path.join(results_dir, "run1-final-results.json")) as f:
    results = json.load(f)

print("\nOptimization Results:")
print(f"Best objective value: {results['bestScore']:.4f}")
print(f"Best parameters:")
for param, value in results['bestParameters'].items():
    print(f"  {param}: {value:.4f}")
```

### Visualize Convergence

If `saveVisualization=True`, convergence plots are in the `plots/` directory:

- `convergence_plot.png` - Objective value vs evaluation number
- `parameter_evolution.png` - How parameters changed during optimization

Open these with an image viewer to see optimization progress.

## Complete Example Script

Here's the complete code all together:

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import json
import os

# Setup
vps.setDimension(2)
vls.setDimension(2)

# Create project
project = fit.Project("myFirstOptimization", "./projects").initialize()

# Create initial domain
gridDelta = 5.0
initialDomain = vps.Domain(
    gridDelta=gridDelta,
    xExtent=400.0,
    yExtent=200.0,
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY
)
substrate = vls.Domain(
    [-200, 200, -100, 100],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(substrate, vls.lsBox([0, -50], [400, 30])).apply()
initialDomain.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)
project.setInitialDomain(initialDomain)

# Create target domain
targetLevelSet = vls.Domain(
    [-200, 200, -100, 100],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(targetLevelSet, vls.lsBox([80, -50], [320, -20])).apply()
project.setTargetLevelSet(targetLevelSet)

# Define process sequence
def processSequence(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    model = vps.IsotropicProcess(rate=-params["etchRate"])
    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(params["processTime"])
    process.apply()
    return domain.getLevelSets()[-1]

# Configure and run optimization
opt = fit.Optimization(project)
opt.setProcessSequence(processSequence)
opt.setParameterNames(["etchRate", "processTime"])
opt.setVariableParameters({
    "etchRate": (5.0, 50.0),
    "processTime": (0.5, 3.0)
})
opt.setDistanceMetrics(primaryMetric="CA")
opt.setName("run1")
opt.apply(numEvaluations=50, saveVisualization=True)

# View results
results_dir = os.path.join(project.projectPath, "optimizationRuns", "run1")
with open(os.path.join(results_dir, "run1-final-results.json")) as f:
    results = json.load(f)

print(f"\nBest objective: {results['bestScore']:.4f}")
print("Best parameters:", results['bestParameters'])
```

## Next Steps

Congratulations! You've run your first ViennaFit optimization.

To learn more:

1. **[Core Concepts](concepts.md)** - Understand projects, domains, metrics, and parameters in depth
2. **[Tutorial 1: Basic Optimization](../tutorials/tutorial-1-basic-optimization.md)** - More detailed walkthrough with real process models
3. **[Tutorial 2: Custom Evaluation](../tutorials/tutorial-2-custom-evaluation.md)** - Explore parameter space beyond optimization
4. **[Tutorial 3: Sensitivity Analysis](../tutorials/tutorial-3-sensitivity-analysis.md)** - Identify which parameters matter most

## Common Next Questions

??? question "How do I choose the right distance metric?"
    Start with **CCH** (Chamfer) for most shape-matching tasks. It's robust and handles both global shape and local features well.

    - Use **CA** for simple area matching (fastest)
    - Use **CCD** when specific dimensions matter (e.g., trench depth, sidewall angle)
    - Use **CSF** for very detailed matching (slowest)

    See [Core Concepts - Distance Metrics](concepts.md#distance-metrics) for detailed comparison.

??? question "How many evaluations do I need?"
    Depends on:

    - **Number of parameters**: More parameters need more evaluations
    - **Parameter landscape**: Complex landscapes need more exploration
    - **Desired accuracy**: Higher accuracy needs more evaluations

    **Rules of thumb:**
    - 2-3 parameters: 50-100 evaluations
    - 4-6 parameters: 100-300 evaluations
    - 7+ parameters: 300-1000+ evaluations

    Monitor convergence plots - if still improving, run more evaluations.

??? question "Can I resume if optimization is interrupted?"
    Yes! ViennaFit 2.0+ supports loading incomplete runs:

    ```python
    evaluator = fit.CustomEvaluator(project)
    evaluator.loadOptimizationRun("run1")  # Works even if incomplete
    best_params = evaluator.loadBestFromProgressCSV("run1")
    # Use best_params for further refinement
    ```

    See [Tutorial 2](../tutorials/tutorial-2-custom-evaluation.md#working-with-incomplete-runs) for details.

??? question "How do I use different optimizers?"
    Change the optimizer with `.setOptimizer()`:

    ```python
    opt.setOptimizer("dlib")       # Default, global optimization
    opt.setOptimizer("nevergrad")  # Evolutionary algorithm
    opt.setOptimizer("ax")         # Bayesian optimization
    ```

    **dlib**: Good default for most cases
    **Nevergrad**: Better for complex landscapes
    **Ax**: Best for expensive simulations (fewer evaluations needed)
