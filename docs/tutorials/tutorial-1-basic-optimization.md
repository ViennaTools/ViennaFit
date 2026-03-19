---
jupytext:
  formats: "docs/tutorials//md,notebooks//ipynb"
  text_representation:
    extension: .md
    format_name: markdown
    format_version: '1.3'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial 1: Basic Optimization

Learn the complete optimization workflow from project setup to result analysis through a realistic plasma etching calibration example.

**Estimated time:** 30-45 minutes

## What You'll Learn

- Create and configure a ViennaFit project
- Load experimental data as target domain
- Define a physics-based process sequence
- Set parameter bounds and choose distance metrics
- Run optimization and interpret results
- Validate and refine your calibration

## Prerequisites

- ViennaFit installed
- Basic Python knowledge
- Familiarity with ViennaPS process models
- Understanding of [Core Concepts](../getting-started/concepts.md)

## Scenario

You have SEM cross-section images of a plasma-etched silicon trench and want to calibrate your process model parameters to match the experimental profile.

**Process:** SF6/C4F8 plasma etching (Bosch process)

**Parameters to calibrate:**
- Ion source power (controls angular distribution)
- Neutral etchant rate
- Ion mean energy
- Neutral sticking probability
- Ion etch rate

**Goal:** Find parameters that make simulated profile match experimental cross-section

## Step 1: Project Setup

Create a new Python script (`etch_calibration.py`):

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import json
import os

# Set dimensions (2D simulation)
vps.setDimension(2)
vls.setDimension(2)

# Create project
project = fit.Project("etchCalibration", "./projects")
project.initialize()

print(f"Project created at: {project.projectPath}")
```

Run it:
```bash
python etch_calibration.py
```

Output:
```
Project created at: ./projects/etchCalibration
```

The directory structure is now ready:
```
projects/etchCalibration/
├── etchCalibration-info.json
├── domains/
│   ├── initialDomain/
│   ├── targetDomain/
│   └── annotations/
├── optimizationRuns/
└── ...
```

## Step 2: Set the Initial Domain for your project

Define the initial substrate geometry before etching. `vps.MakeTrench` with `makeMask=True`
creates a silicon substrate with a patterned mask (opening of `trenchWidth`) on top — no trench
cut into the substrate yet:

```python
# Grid resolution
gridDelta = 5.0  # nm

initialDomain = vps.Domain()
vps.MakeTrench(
    initialDomain,
    gridDelta=gridDelta,
    xExtent=600.0,       # 600nm wide
    yExtent=400.0,       # 400nm tall
    trenchWidth=200.0,   # mask opening width (x=-100 to x=100)
    trenchDepth=50.0,    # mask height (nm)
    makeMask=True,       # substrate + mask with opening; no trench yet
    material=vps.Material.Si
).apply()

project.setInitialDomain(initialDomain)
initialDomain.saveSurfaceMesh(
    os.path.join(project.projectPath, "domains/initialDomain/initial.vtp"),
    True  # Add material IDs
)

print("Initial domain created and saved")
```

!!! tip "Visualizing Domains"
    Open `.vtp` files in ParaView or VisIt to visualize the geometry.

## Step 3: Load Target Domain

In a real scenario, you'd load experimental data. For this tutorial, we'll create a target that represents a typical etched trench. `vps.MakeTrench` with `makeMask=False` (default) creates a silicon substrate with a trench already etched into it:

```python
targetDomain = vps.Domain()
vps.MakeTrench(
    targetDomain,
    gridDelta=gridDelta,
    xExtent=600.0,       # 600nm wide
    yExtent=400.0,       # 400nm tall
    trenchWidth=200.0,   # trench width (x=-100 to x=100)
    trenchDepth=150.0,   # trench depth (150nm)
    material=vps.Material.Si
).apply()

project.setTargetLevelSet(targetDomain.getLevelSets()[-1])

targetMesh = vls.Mesh()
vls.ToSurfaceMesh(targetDomain.getLevelSets()[-1], targetMesh).apply()
writer = vls.VTKWriter(targetMesh)
writer.setFileName(
    os.path.join(project.projectPath, "domains/targetDomain/target.vtp")
)
writer.apply()

print("Target domain created and saved")
```

!!! info "Real Experimental Data"
    To load actual experimental data:
    ```python
    target = vls.Domain([...], boundary, gridDelta)
    reader = vls.VTKReader(target)
    reader.apply("experimental_profile.vtp")
    project.setTargetLevelSet(target)
    ```

## Step 4: Define Process Sequence

Create a physics-based etching process:

```python
def etchingProcess(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    """
    SF6/C4F8 plasma etching process.

    Parameters:
    - ionSourcePower: Source power controlling ion angular distribution
    - neutralRate: Neutral etchant etch rate contribution (nm/s)
    - ionEnergy: Ion mean energy (eV)
    - neutralStickP: Neutral sticking probability
    - ionRate: Ion etch rate contribution (nm/s)
    """
    # Set up model
    model = vps.MultiParticleProcess()

    # Add neutral etchant particles first (→ fluxes[0])
    sticking = {
        vps.Material.Si: params["neutralStickP"],
        vps.Material.SiO2: 0.01  # Low sticking on mask
    }
    model.addNeutralParticle(
        sticking=sticking,
        label="etchant"
    )

    # Add ion particles second (→ fluxes[1])
    model.addIonParticle(
        sourcePower=params["ionSourcePower"],
        meanEnergy=params["ionEnergy"],
        sigmaEnergy=10.0,  # Energy spread
        label="ion"
    )

    # Define etch rate function
    # fluxes[0] = neutral flux, fluxes[1] = ion flux
    # Rates are negative for etching (material removal)
    def rateFunction(fluxes, material):
        if material == vps.Material.Si:
            # Silicon etches with both ions and neutrals
            neutralContrib = -fluxes[0] * params["neutralRate"]
            ionContrib = -fluxes[1] * params["ionRate"]
            return neutralContrib + ionContrib
        elif material == vps.Material.SiO2:
            # Mask etches much slower (ion-only)
            return -fluxes[1] * 0.05
        return 0.0

    model.setRateFunction(rateFunction)

    # Run process
    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(1.0)  # 1 second

    # Use CPU disk flux engine for accurate flux calculation
    process.setFluxEngineType(vps.FluxEngineType.CPU_DISK)

    # Apply
    process.apply()

    # Planarize at y=50 to clean up top surface before comparing
    vps.Planarize(domain, 50.0).apply()

    # Return resulting level set (etched substrate is the last level set)
    return domain.getLevelSets()[-1]
```

!!! note "Process Complexity"
    This is a simplified model. Real processes may include:
    - Deposition/etching cycles (Bosch process)
    - Surface reactions
    - Multiple particle species
    - Temperature effects

## Step 5: Configure Optimization

Set up the optimization with parameters and distance metric:

```python
# Create optimization
opt = fit.Optimization(project)
opt.setProcessSequence(etchingProcess)

# Define parameter names
opt.setParameterNames([
    "ionSourcePower",
    "neutralRate",
    "ionEnergy",
    "neutralStickP",
    "ionRate"
])

# Set parameter bounds (based on physical constraints and literature)
opt.setVariableParameters({
    "ionSourcePower": (10.0, 200.0),   # source power (controls angular spread)
    "neutralRate":    (0.001, 20.0),   # neutral etch rate (nm/s)
    "ionEnergy":      (20.0, 200.0),   # eV
    "neutralStickP":  (0.01, 0.9),     # probability
    "ionRate":        (0.01, 20.0)     # ion etch rate (nm/s)
})

# Choose distance metric
# CCH (Chamfer) is good for shape matching
opt.setDistanceMetrics(
    primaryMetric="CCH",
    additionalMetrics=["CA"]  # Also track area for reference
)

# Name and document this run
opt.setName("run1_initialCalibration")
opt.setNotes(
    "Initial calibration of SF6/C4F8 etching parameters. "
    "Using CCH metric to match trench profile shape. "
    "100 evaluations with dlib optimizer."
)

print("Optimization configured")
```

### Choosing Parameter Bounds

**How to set good bounds:**

1. **Check literature** for typical values
2. **Start wide**, narrow after initial run
3. **Physical constraints**: e.g., probabilities must be 0-1
4. **Experimental conditions**: match your setup

**Our bounds rationale:**
- `ionSourcePower`: Controls the angular distribution of ions; 10-200 covers typical plasma conditions
- `neutralRate`: Neutral etch rate contribution; 0.001-20 nm/s spans slow to aggressive chemistries
- `ionEnergy`: Typical RF plasma energies 20-200 eV
- `neutralStickP`: Must be 0-1, typical range 0.01-0.9
- `ionRate`: Ion etch rate contribution; 0.01-20 nm/s

## Step 6: Run Optimization

Execute the optimization:

```python
print("\n" + "="*60)
print("STARTING OPTIMIZATION")
print("="*60)

opt.apply(
    numEvaluations=100,         # Try 100 parameter combinations
    saveComparison=True      # Save intermediate results
)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)
```

**What happens:**
1. dlib chooses initial parameters
2. For each evaluation:
   - Copy initial domain
   - Run `etchingProcess` with chosen parameters
   - Compare result to target using CCH metric
   - Record objective value
3. dlib analyzes results and chooses next parameters
4. Repeat until 100 evaluations or convergence

**Console output (example):**
```
==============================================================
STARTING OPTIMIZATION
==============================================================
Evaluation 1/100: Objective = 145.67
  ionSourcePower: 105.2, neutralRate: 8.3, ionEnergy: 110.5, neutralStickP: 0.45, ionRate: 7.1
Evaluation 2/100: Objective = 132.89
  ionSourcePower: 88.7, neutralRate: 11.1, ionEnergy: 95.2, neutralStickP: 0.38, ionRate: 5.4
...
Evaluation 97/100: Objective = 23.41
  ionSourcePower: 72.3, neutralRate: 6.5, ionEnergy: 88.7, neutralStickP: 0.31, ionRate: 3.9
...
==============================================================
OPTIMIZATION COMPLETE
==============================================================
Best objective value: 21.87
Best parameters:
  ionSourcePower: 74.1
  neutralRate: 6.9
  ionEnergy: 85.3
  neutralStickP: 0.29
  ionRate: 4.2
```

**Runtime:** 15-60 minutes depending on simulation complexity.

## Step 7: Analyze Results

### Load and Display Results

```python
# Results location
results_dir = os.path.join(
    project.projectPath,
    "optimizationRuns",
    "run1_initialCalibration"
)

# Load best parameters
with open(os.path.join(results_dir, "run1_initialCalibration-final-results.json")) as f:
    results = json.load(f)

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"\nBest Objective Value: {results['bestScore']:.4f}")
print(f"Achieved at Evaluation: {results['bestEvaluation#']}")
print("\nBest Parameters:")
for param, value in results['bestParameters'].items():
    print(f"  {param:20s}: {value:.4f}")

if 'fixedParameters' in results:
    print("\nFixed Parameters:")
    for param, value in results['fixedParameters'].items():
        print(f"  {param:20s}: {value:.4f}")
```

### View Convergence Plot

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load progress data
progress = pd.read_csv(os.path.join(results_dir, "progressBest.csv"))

# Plot objective value over time
plt.figure(figsize=(10, 6))
plt.plot(progress['evaluationNumber'], progress['objectiveValue'], 'b-', linewidth=2)
plt.xlabel('Evaluation Number')
plt.ylabel('Objective Value (CCH Distance)')
plt.title('Optimization Convergence')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, "convergence_analysis.png"), dpi=150)
print(f"\nConvergence plot saved to: {results_dir}/convergence_analysis.png")
```

### Interpret Convergence

Good convergence shows:

✅ **Steady decrease** in objective value
✅ **Plateau** at the end (converged)
✅ **Few evaluations** at plateau (efficient)

Bad convergence shows:

❌ **Still decreasing** at end (needs more evaluations)
❌ **Erratic jumps** (noisy objective, bad parameters)
❌ **No improvement** (poor initial bounds)

### Parameter Evolution

See how parameters changed:

```python
# Load all evaluations
all_evals = pd.read_csv(os.path.join(results_dir, "progressAll.csv"))

# Plot parameter evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
params = ["ionSourcePower", "neutralRate", "ionEnergy", "neutralStickP", "ionRate"]

for i, param in enumerate(params):
    ax = axes[i//2, i%2]
    ax.plot(all_evals['evaluationNumber'], all_evals[param], 'o-', alpha=0.6)
    ax.set_xlabel('Evaluation Number')
    ax.set_ylabel(param)
    ax.set_title(f'{param} Evolution')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "parameter_evolution.png"), dpi=150)
print(f"Parameter evolution plot saved")
```

## Step 8: Validate Results

### Re-run with Best Parameters

Verify the best parameters actually produce good results:

```python
# Get best parameters
best = results['bestParameters']

# Re-run simulation
validation_domain = vps.Domain(initialDomain)  # Copy initial domain
result_ls = etchingProcess(validation_domain, best)

# Save result
validation_mesh = vls.Mesh()
vls.ToSurfaceMesh(result_ls, validation_mesh).apply()
writer = vls.VTKWriter(validation_mesh)
writer.setFileName(os.path.join(results_dir, "validation_best_result.vtp"))
writer.apply()

print(f"\nValidation result saved to: {results_dir}/validation_best_result.vtp")
print("Open in ParaView to compare with target")
```

### Visual Comparison

Open in ParaView:
1. Load `target.vtp` (from domains/targetDomain/)
2. Load `validation_best_result.vtp`
3. Display both with different colors
4. Check alignment

### Quantitative Validation

Compute additional metrics:

```python
# Re-compute distance metrics
from viennafit.fitDistanceMetrics import DistanceMetric

# Create metric computer
metric_CCH = DistanceMetric.create("CCH", False)
metric_CA = DistanceMetric.create("CA", False)

# Load target
target_ls = project.targetLevelSet  # or reload

# Compute metrics
cch_value = metric_CCH(result_ls, target_ls)
ca_value = metric_CA(result_ls, target_ls)

print(f"\nValidation Metrics:")
print(f"  CCH (Chamfer): {cch_value:.4f}")
print(f"  CA (Area):     {ca_value:.4f}")
print(f"\n(Should match optimization best score: {results['bestScore']:.4f})")
```

## Step 9: Refine if Needed

### If Results Unsatisfactory

**Option 1: More Evaluations**
```python
# Continue optimization
opt2 = fit.Optimization(project)
# ... configure same as before ...
opt.setName("run2_continued")
# Use run1 results as starting point
opt.apply(numEvaluations=50, saveComparison=True)
```

**Option 2: Narrow Bounds**
```python
# Focus search around best values
best = results['bestParameters']
opt.setVariableParameters({
    "ionSourcePower": (best["ionSourcePower"] * 0.9, best["ionSourcePower"] * 1.1),
    "neutralRate":    (best["neutralRate"] * 0.9,    best["neutralRate"] * 1.1),
    # ... etc
})
```

**Option 3: Different Metric**
```python
# Try different distance metric
opt.setDistanceMetrics(primaryMetric="CCD",  # Critical dimensions
    criticalDimensionRanges=[
        {'axis': 'y', 'min': 100, 'max': 500, 'findMaximum': True}  # Trench depth
    ])
```

### If Results Good

Move to sensitivity analysis or parameter exploration:
- [Tutorial 2: Custom Evaluation](tutorial-2-custom-evaluation.md)
- [Tutorial 3: Sensitivity Analysis](tutorial-3-sensitivity-analysis.md)

## Complete Script

Here's the full script for reference:

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
project = fit.Project("etchCalibration", "./projects")
project.initialize()

# ── Step 2: Initial Domain ──────────────────────────────────────────────────
gridDelta = 5.0

initialDomain = vps.Domain()
vps.MakeTrench(
    initialDomain,
    gridDelta=gridDelta,
    xExtent=600.0,
    yExtent=400.0,
    trenchWidth=200.0,
    trenchDepth=50.0,
    makeMask=True,
    material=vps.Material.Si
).apply()
project.setInitialDomain(initialDomain)
initialDomain.saveSurfaceMesh(
    os.path.join(project.projectPath, "domains/initialDomain/initial.vtp"), True
)

# ── Step 3: Target Domain ───────────────────────────────────────────────────
targetDomain = vps.Domain()
vps.MakeTrench(
    targetDomain,
    gridDelta=gridDelta,
    xExtent=600.0,
    yExtent=400.0,
    trenchWidth=200.0,
    trenchDepth=150.0,
    material=vps.Material.Si
).apply()
project.setTargetLevelSet(targetDomain.getLevelSets()[-1])
targetMesh = vls.Mesh()
vls.ToSurfaceMesh(targetDomain.getLevelSets()[-1], targetMesh).apply()
writer = vls.VTKWriter(targetMesh)
writer.setFileName(os.path.join(project.projectPath, "domains/targetDomain/target.vtp"))
writer.apply()

# ── Step 4: Process Sequence ────────────────────────────────────────────────
def etchingProcess(domain: vps.Domain, params: dict[str, float]) -> vls.Domain:
    model = vps.MultiParticleProcess()

    # Neutral etchant particles first (→ fluxes[0])
    sticking = {
        vps.Material.Si: params["neutralStickP"],
        vps.Material.SiO2: 0.01
    }
    model.addNeutralParticle(sticking=sticking, label="etchant")

    # Ion particles second (→ fluxes[1])
    model.addIonParticle(
        sourcePower=params["ionSourcePower"],
        meanEnergy=params["ionEnergy"],
        sigmaEnergy=10.0,
        label="ion"
    )

    # Rates are negative for material removal
    def rateFunction(fluxes, material):
        if material == vps.Material.Si:
            return -fluxes[0] * params["neutralRate"] - fluxes[1] * params["ionRate"]
        elif material == vps.Material.SiO2:
            return -fluxes[1] * 0.05
        return 0.0

    model.setRateFunction(rateFunction)

    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(1.0)
    process.setFluxEngineType(vps.FluxEngineType.CPU_DISK)
    process.apply()

    vps.Planarize(domain, 50.0).apply()
    return domain.getLevelSets()[-1]

# ── Step 5: Configure Optimization ─────────────────────────────────────────
opt = fit.Optimization(project)
opt.setProcessSequence(etchingProcess)
opt.setParameterNames(["ionSourcePower", "neutralRate", "ionEnergy", "neutralStickP", "ionRate"])
opt.setVariableParameters({
    "ionSourcePower": (10.0, 200.0),
    "neutralRate":    (0.001, 20.0),
    "ionEnergy":      (20.0, 200.0),
    "neutralStickP":  (0.01, 0.9),
    "ionRate":        (0.01, 20.0)
})
opt.setDistanceMetrics(primaryMetric="CCH", additionalMetrics=["CA"])
opt.setName("run1_initialCalibration")

# ── Step 6: Run ─────────────────────────────────────────────────────────────
opt.apply(numEvaluations=100, saveComparison=True)

# ── Step 7: Analyze ─────────────────────────────────────────────────────────
results_dir = os.path.join(project.projectPath, "optimizationRuns", "run1_initialCalibration")
with open(os.path.join(results_dir, "run1_initialCalibration-final-results.json")) as f:
    results = json.load(f)

print(f"Best Score: {results['bestScore']:.4f}")
print("Best Parameters:", results['bestParameters'])
```

## Key Takeaways

✅ **Projects organize everything** - All results in one place
✅ **Process sequences are flexible** - Can be as complex as needed
✅ **Choose appropriate metrics** - CCH good for shape, CA for area
✅ **Monitor convergence** - Check plots to see if converged
✅ **Validate results** - Re-run with best parameters to verify

## Common Issues

??? question "Optimization not converging"
    **Symptoms:** Objective still decreasing at end

    **Solutions:**
    - Increase `numEvaluations`
    - Check parameter bounds aren't too wide
    - Try different optimizer (`opt.setOptimizer("nevergrad")`)

??? question "Results don't match target"
    **Symptoms:** Low objective value but visual mismatch

    **Solutions:**
    - Wrong distance metric for your goal (try CCH instead of CA)
    - Process model too simple (add more physics)
    - Target domain incorrectly loaded

??? question "Simulation crashes"
    **Symptoms:** Process sequence throws errors

    **Solutions:**
    - Check domain bounds are reasonable
    - Verify material types match
    - Add error handling in process sequence

## Next Steps

**Continue learning:**
- [Tutorial 2: Custom Parameter Evaluation](tutorial-2-custom-evaluation.md) - Explore parameter space
- [Tutorial 3: Sensitivity Analysis](tutorial-3-sensitivity-analysis.md) - Find important parameters
- [Tutorial 4: Multi-Domain Optimization](tutorial-4-multi-domain.md) - Universal parameters

**Improve your calibration:**
- Use real experimental data as target
- Add more physical effects to process model
- Try Bayesian optimization for expensive simulations
- Perform sensitivity analysis to reduce parameters
