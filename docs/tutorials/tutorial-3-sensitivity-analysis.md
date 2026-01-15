# Tutorial 3: Sensitivity Analysis

Learn how to identify which parameters matter most using local and global sensitivity analysis methods.

**Estimated time:** 30-40 minutes

## What You'll Learn

- Perform local sensitivity analysis (one-at-a-time)
- Conduct global sensitivity analysis (Sobol indices)
- Interpret sensitivity results
- Use results to simplify optimization problems
- Determine which parameters need tighter control

## Prerequisites

- Completed [Tutorial 1](tutorial-1-basic-optimization.md)
- Understanding of [Core Concepts](../getting-started/concepts.md)
- Familiarity with basic statistics

## Why Sensitivity Analysis?

**Sensitivity analysis answers:**
- Which parameters significantly affect the output?
- Which parameters can be fixed without hurting performance?
- How do parameters interact with each other?
- Which parameters need precise control in manufacturing?

**Use cases:**
- **Model reduction**: Fix insensitive parameters
- **Experimental design**: Focus measurements on sensitive parameters
- **Uncertainty quantification**: Prioritize parameters for tighter bounds
- **Process control**: Identify critical control points

## Scenario

After optimizing etch process parameters, you want to know:
1. Which parameters have the biggest impact on trench profile?
2. Can you fix some parameters to simplify the problem?
3. Do parameters interact (e.g., ion flux effect depends on energy)?

## Part 1: Local Sensitivity Analysis

### Step 1: Setup

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Setup
vps.setDimension(2)
vls.setDimension(2)

# Load project
project = fit.Project().load("./projects/etchCalibration")

# Create local sensitivity study
local_study = fit.LocalSensitivityStudy(
    name="localSens1",
    project=project
)

print("Local sensitivity study initialized")
```

### Step 2: Load Process Sequence and Parameters

```python
# Load process sequence from optimization run
process_seq_path = (
    "./projects/etchCalibration/optimizationRuns/"
    "run1_initialCalibration/run1_initialCalibration-processSequence.py"
)
local_study.loadProcessSequence(process_seq_path)

# Define parameters to analyze
local_study.setParameterNames([
    "ionFlux",
    "etchantFlux",
    "ionEnergy",
    "neutralStickP"
])

print("Process sequence and parameters loaded")
```

### Step 3: Define Sensitivity Ranges

Local sensitivity varies one parameter at a time around a baseline:

```python
# Get optimal parameters from optimization as baseline
import json
with open("./projects/etchCalibration/optimizationRuns/"
          "run1_initialCalibration/run1_initialCalibration-final-results.json") as f:
    results = json.load(f)

optimal = results['bestParameters']

# Define ranges: (low, baseline, high) for each parameter
# Vary by ±20% around optimal
local_study.setParameterSensitivityRanges({
    "ionFlux": (
        optimal["ionFlux"] * 0.8,
        optimal["ionFlux"],
        optimal["ionFlux"] * 1.2
    ),
    "etchantFlux": (
        optimal["etchantFlux"] * 0.8,
        optimal["etchantFlux"],
        optimal["etchantFlux"] * 1.2
    ),
    "ionEnergy": (
        optimal["ionEnergy"] * 0.8,
        optimal["ionEnergy"],
        optimal["ionEnergy"] * 1.2
    ),
    "neutralStickP": (
        optimal["neutralStickP"] * 0.8,
        optimal["neutralStickP"],
        optimal["neutralStickP"] * 1.2
    )
})

print(f"\nSensitivity ranges defined around optimal:")
for param, value in optimal.items():
    print(f"  {param}: {value:.4f} ± 20%")
```

### Step 4: Configure and Run

```python
# Set distance metric (same as optimization)
local_study.setDistanceMetric("CCH")

# Run local sensitivity analysis
# nEval=(5,) means 5 points per parameter
print("\nRunning local sensitivity analysis...")
local_study.apply(nEval=(5,))  # Total: 4 params × 5 points = 20 evaluations

print("Local sensitivity analysis complete!")
```

**Runtime:** 5-20 minutes depending on simulation complexity

### Step 5: Analyze Results

```python
# Load results
results_dir = f"{project.projectPath}/localSensitivityStudies/localSens1"

# Load CSV data
csv_path = f"{results_dir}/localSens1_results.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    print("\nLocal Sensitivity Results:")
    print(df.head())

    # Plot sensitivity for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    params = ["ionFlux", "etchantFlux", "ionEnergy", "neutralStickP"]

    for i, param in enumerate(params):
        ax = axes[i//2, i%2]

        # Filter data for this parameter (others at baseline)
        param_data = df[df['variableParameter'] == param]

        if len(param_data) > 0:
            ax.plot(param_data[param], param_data['objectiveValue'],
                   'o-', linewidth=2, markersize=8)
            ax.set_xlabel(param)
            ax.set_ylabel('Objective Value')
            ax.set_title(f'Sensitivity to {param}')
            ax.grid(True, alpha=0.3)

            # Mark baseline
            baseline_val = optimal[param]
            ax.axvline(baseline_val, color='r', linestyle='--',
                      label='Baseline')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_dir}/local_sensitivity_plots.png", dpi=150)
    print(f"\nSensitivity plots saved to {results_dir}/")
```

### Step 6: Compute Sensitivity Indices

```python
# Compute local sensitivity (numerical derivative)
sensitivities = {}

for param in params:
    param_data = df[df['variableParameter'] == param].sort_values(param)

    if len(param_data) >= 3:
        # Compute derivative around baseline
        param_vals = param_data[param].values
        obj_vals = param_data['objectiveValue'].values

        # Central difference approximation
        baseline_idx = len(param_vals) // 2  # Middle point
        if baseline_idx > 0 and baseline_idx < len(param_vals) - 1:
            delta_param = param_vals[baseline_idx+1] - param_vals[baseline_idx-1]
            delta_obj = obj_vals[baseline_idx+1] - obj_vals[baseline_idx-1]
            sensitivity = abs(delta_obj / delta_param)
            sensitivities[param] = sensitivity

# Normalize sensitivities
total_sens = sum(sensitivities.values())
norm_sens = {k: v/total_sens for k, v in sensitivities.items()}

print("\nNormalized Local Sensitivities:")
for param, sens in sorted(norm_sens.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param:20s}: {sens*100:.2f}%")

# Plot ranking
plt.figure(figsize=(10, 6))
sorted_params = sorted(norm_sens.items(), key=lambda x: x[1], reverse=True)
plt.barh([p[0] for p in sorted_params], [p[1]*100 for p in sorted_params])
plt.xlabel('Relative Sensitivity (%)')
plt.title('Local Parameter Sensitivity Ranking')
plt.tight_layout()
plt.savefig(f"{results_dir}/sensitivity_ranking.png", dpi=150)
print(f"\nSensitivity ranking plot saved")
```

### Interpret Local Results

**High sensitivity parameters:**
- Large change in objective with small parameter change
- Need precise control
- Cannot be fixed without hurting performance

**Low sensitivity parameters:**
- Small change in objective even with large parameter variation
- Can potentially be fixed at nominal value
- Less critical for process control

## Part 2: Global Sensitivity Analysis

Global sensitivity uses Sobol sampling to explore the entire parameter space and compute variance-based sensitivity indices.

### Step 1: Setup Global Study

```python
# Create global sensitivity study
global_study = fit.GlobalSensitivityStudy(
    name="globalSens1",
    project=project
)

# Load process sequence
global_study.loadProcessSequence(process_seq_path)

# Define parameters
global_study.setParameterNames([
    "ionFlux",
    "etchantFlux",
    "ionEnergy",
    "neutralStickP"
])

print("Global sensitivity study initialized")
```

### Step 2: Define Parameter Bounds

Use the full parameter space, not just around optimum:

```python
# Use same bounds as optimization
global_study.setVariableParameters({
    "ionFlux": (10.0, 100.0),
    "etchantFlux": (100.0, 1000.0),
    "ionEnergy": (20.0, 200.0),
    "neutralStickP": (0.01, 0.9)
})

print("\nGlobal parameter bounds set")
```

### Step 3: Configure Sampling

```python
# Configure Sobol sampling
global_study.setSamplingOptions(
    numSamples=128,        # Base sample size
    secondOrder=True       # Compute interaction effects
)

# Total evaluations = numSamples * (2 * numParams + 2)
# For 4 params: 128 * (2*4 + 2) = 128 * 10 = 1280 evaluations

# Set distance metric
global_study.setDistanceMetric("CCH")

print(f"Sampling configured: ~1280 evaluations")
print("⚠ This will take significant time (1-4 hours)")
```

!!! warning "Computational Cost"
    Global sensitivity is expensive! Consider:
    - Start with fewer samples (64) for testing
    - Use simpler distance metric (CA) initially
    - Run overnight or on cluster
    - Or use local sensitivity if budget limited

### Step 4: Run Global Analysis

```python
print("\nRunning global sensitivity analysis...")
print("This will take considerable time. Consider running overnight.")

# Run
global_study.apply()

print("Global sensitivity analysis complete!")
```

### Step 5: Analyze Sobol Indices

```python
# Load results
results_dir_global = f"{project.projectPath}/globalSensitivityStudies/globalSens1"

# Sobol indices from SALib are typically saved in JSON format
sobol_file = f"{results_dir_global}/globalSens1_sobol_indices.json"

if os.path.exists(sobol_file):
    with open(sobol_file) as f:
        sobol = json.load(f)

    print("\nSobol Sensitivity Indices:")
    print("\n First-Order Indices (S1) - Direct effect:")
    for i, param in enumerate(params):
        if param in sobol['S1']:
            print(f"  {param:20s}: {sobol['S1'][param]:.4f}")

    print("\n Total-Order Indices (ST) - Total effect including interactions:")
    for i, param in enumerate(params):
        if param in sobol['ST']:
            print(f"  {param:20s}: {sobol['ST'][param]:.4f}")

    # Plot Sobol indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # S1 plot
    s1_vals = [sobol['S1'][p] for p in params]
    ax1.barh(params, s1_vals)
    ax1.set_xlabel('First-Order Index (S1)')
    ax1.set_title('Direct Parameter Effects')
    ax1.grid(True, alpha=0.3)

    # ST plot
    st_vals = [sobol['ST'][p] for p in params]
    ax2.barh(params, st_vals, color='orange')
    ax2.set_xlabel('Total-Order Index (ST)')
    ax2.set_title('Total Effects (incl. interactions)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir_global}/sobol_indices.png", dpi=150)
    print(f"\nSobol indices plot saved")

    # Compute interaction strength
    print("\n Interaction Effects (ST - S1):")
    for i, param in enumerate(params):
        interaction = sobol['ST'][param] - sobol['S1'][param]
        print(f"  {param:20s}: {interaction:.4f}")
```

### Interpret Global Results

**Sobol Indices Interpretation:**

- **S1 (First-Order)**: Direct effect of parameter alone
  - S1 = 0.6 means 60% of output variance due to this parameter

- **ST (Total-Order)**: Total effect including all interactions
  - ST = 0.8 means 80% of variance when including interactions

- **ST - S1**: Interaction effects
  - Large difference → parameter interacts strongly with others
  - Small difference → parameter acts independently

**Decision rules:**
- S1 > 0.1: Parameter is important, keep in optimization
- S1 < 0.01: Parameter is negligible, consider fixing
- (ST - S1) > 0.2: Strong interactions present

## Step 7: Use Results to Simplify Model

Based on sensitivity analysis, identify parameters to fix:

```python
# Assume global sensitivity found:
# - ionFlux: S1 = 0.45 (important)
# - etchantFlux: S1 = 0.35 (important)
# - ionEnergy: S1 = 0.15 (moderate)
# - neutralStickP: S1 = 0.03 (negligible)

print("\nParameter Classification:")
print("  Critical (S1 > 0.3):")
print("    - ionFlux")
print("    - etchantFlux")
print("  Moderate (0.1 < S1 < 0.3):")
print("    - ionEnergy")
print("  Negligible (S1 < 0.1):")
print("    - neutralStickP")

print("\nRecommendation:")
print("  Fix neutralStickP at optimal value")
print("  Optimize remaining 3 parameters")
print("  This reduces search space by 25%!")
```

### Re-run Optimization with Simplified Model

```python
# Create new optimization with fewer parameters
opt_simplified = fit.Optimization(project)
opt_simplified.loadProcessSequence(process_seq_path)

# Only optimize sensitive parameters
opt_simplified.setParameterNames([
    "ionFlux",
    "etchantFlux",
    "ionEnergy"
])

opt_simplified.setVariableParameters({
    "ionFlux": (10.0, 100.0),
    "etchantFlux": (100.0, 1000.0),
    "ionEnergy": (20.0, 200.0)
})

# Fix insensitive parameter
opt_simplified.setFixedParameters({
    "neutralStickP": optimal["neutralStickP"]  # Fixed at optimal
})

opt_simplified.setDistanceMetrics(primaryMetric="CCH")
opt_simplified.setName("run2_simplified")

print("\nReady to run simplified optimization (3 parameters instead of 4)")
# opt_simplified.apply(numEvaluations=50)  # Faster convergence!
```

## Complete Example Script

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import json

# Setup
vps.setDimension(2)
vls.setDimension(2)
project = fit.Project().load("./projects/etchCalibration")

# Get optimal parameters
with open("./projects/etchCalibration/optimizationRuns/"
          "run1_initialCalibration/run1_initialCalibration-final-results.json") as f:
    optimal = json.load(f)['bestParameters']

# === Local Sensitivity ===
local_study = fit.LocalSensitivityStudy("localSens1", project)
local_study.loadProcessSequence("path/to/processSequence.py")
local_study.setParameterNames(list(optimal.keys()))

# Define ranges (±20% around optimal)
ranges = {param: (val*0.8, val, val*1.2) for param, val in optimal.items()}
local_study.setParameterSensitivityRanges(ranges)

local_study.setDistanceMetric("CCH")
local_study.apply(nEval=(5,))

# === Global Sensitivity ===
global_study = fit.GlobalSensitivityStudy("globalSens1", project)
global_study.loadProcessSequence("path/to/processSequence.py")
global_study.setParameterNames(list(optimal.keys()))
global_study.setVariableParameters({
    "ionFlux": (10.0, 100.0),
    "etchantFlux": (100.0, 1000.0),
    "ionEnergy": (20.0, 200.0),
    "neutralStickP": (0.01, 0.9)
})
global_study.setSamplingOptions(numSamples=128, secondOrder=True)
global_study.setDistanceMetric("CCH")

# global_study.apply()  # Long runtime!
```

## Key Takeaways

✅ **Local sensitivity** - Fast, one-at-a-time, around specific point
✅ **Global sensitivity** - Comprehensive, entire space, finds interactions
✅ **Use Sobol indices** - S1 for direct effects, ST for total effects
✅ **Simplify models** - Fix insensitive parameters
✅ **Prioritize efforts** - Focus on sensitive parameters

## Common Questions

??? question "Local vs Global - which to use?"
    **Local sensitivity:**
    - Fast (20-100 evaluations)
    - Good for understanding around optimum
    - Use when computational budget limited

    **Global sensitivity:**
    - Expensive (1000+ evaluations)
    - Comprehensive, finds interactions
    - Use when you need full understanding

??? question "How many samples for global sensitivity?"
    **Guidelines:**
    - Minimum: 64 samples
    - Good: 128 samples
    - Excellent: 512+ samples
    - Total evals = samples × (2×params + 2)

??? question "What if all parameters seem important?"
    **Possible reasons:**
    - Parameters truly all matter
    - Parameter bounds too wide
    - Wrong distance metric
    - Process physics requires all

    **Try:** Narrow bounds around physically realistic values

## Next Steps

**Continue learning:**
- [Tutorial 4: Multi-Domain Optimization](tutorial-4-multi-domain.md) - Universal parameters

**Advanced topics:**
- Morris screening (cheaper than Sobol)
- Regional sensitivity analysis
- Metamodel-based sensitivity
- Time-dependent sensitivity
