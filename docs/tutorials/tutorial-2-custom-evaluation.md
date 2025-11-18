# Tutorial 2: Custom Parameter Evaluation

Learn how to explore parameter space, test specific combinations, and perform repeatability analysis using the CustomEvaluator.

**Estimated time:** 25-35 minutes

## What You'll Learn

- Load optimization results for further analysis
- Set up and run parameter grid evaluations
- Test specific parameter combinations
- Perform repeatability testing (variance analysis)
- Work with incomplete optimization runs
- Analyze and visualize results

## Prerequisites

- Completed [Tutorial 1](tutorial-1-basic-optimization.md) or have an existing optimization run
- Understanding of [Core Concepts - Custom Evaluation](../getting-started/concepts.md#custom-evaluation)

## Scenario

After completing an optimization, you want to:
1. Explore the parameter space around the optimum
2. Test some specific parameter combinations you suspect might work well
3. Assess simulation repeatability by running identical parameters multiple times

## Step 1: Load Optimization Results

Start by loading results from a previous optimization:

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set dimensions
vps.setDimension(2)
vls.setDimension(2)

# Load project
project = fit.Project()
project.load("./projects/etchCalibration")

# Create custom evaluator
evaluator = fit.CustomEvaluator(project)

# Load optimization run
evaluator.loadOptimizationRun("run1_initialCalibration")

print("Optimization run loaded successfully")
print(f"Multi-domain mode: {evaluator.isMultiDomainProcess}")
print(f"Optimal parameters: {evaluator.getOptimalParameters()}")
```

**What happened:**
- Loaded project structure
- Created CustomEvaluator instance
- Loaded process sequence and best parameters from optimization
- Ready to evaluate new parameter combinations

!!! info "Loading Incomplete Runs"
    If your optimization was interrupted, ViennaFit 2.0+ can still load it:
    ```python
    evaluator.loadOptimizationRun("incomplete_run")
    # Will print warning but load best parameters from CSV
    ```

## Step 2: Parameter Grid Evaluation

Systematically explore parameter combinations around the optimum:

```python
# Get optimal parameters from optimization
optimal = evaluator.getOptimalParameters()

print("\nOptimal parameters from optimization:")
for param, value in optimal.items():
    print(f"  {param}: {value:.4f}")

# Define grid around optimum
# Vary each parameter by ±20% in 5 steps
evaluator.setVariableValues({
    "ionFlux": [
        optimal["ionFlux"] * 0.8,
        optimal["ionFlux"] * 0.9,
        optimal["ionFlux"] * 1.0,  # Optimal
        optimal["ionFlux"] * 1.1,
        optimal["ionFlux"] * 1.2
    ],
    "etchantFlux": [
        optimal["etchantFlux"] * 0.8,
        optimal["etchantFlux"] * 0.9,
        optimal["etchantFlux"] * 1.0,  # Optimal
        optimal["etchantFlux"] * 1.1,
        optimal["etchantFlux"] * 1.2
    ]
})

# Other parameters fixed at optimal
evaluator.setFixedParameters({
    "ionEnergy": optimal["ionEnergy"],
    "neutralStickP": optimal["neutralStickP"]
})

# Set distance metric (same as optimization)
evaluator.setDistanceMetric("CCH")

print(f"\nGrid defined: 5 × 5 = 25 evaluations")
```

### Run Grid Evaluation

```python
# Execute grid evaluation
results = evaluator.evaluateGrid(
    evaluationName="parameterGrid_ionAndEtchant",
    saveVisualization=False  # Don't save all 25 VTP files
)

print(f"\nGrid evaluation complete!")
print(f"Total evaluations: {len(results)}")
```

**Runtime:** 5-15 minutes for 25 evaluations

### Analyze Grid Results

```python
# Find best result from grid
best_grid = evaluator.getBestResult()

print(f"\nBest result from grid:")
print(f"  Objective value: {best_grid['objectiveValue']:.4f}")
print(f"  Parameters:")
for param, value in best_grid['parameters'].items():
    print(f"    {param}: {value:.4f}")

# Compare to optimization optimum
opt_score = min(r['objectiveValue'] for r in results
                if r['parameters'] == optimal)
print(f"\nOptimization optimum score: {opt_score:.4f}")
print(f"Grid best score: {best_grid['objectiveValue']:.4f}")
print(f"Improvement: {opt_score - best_grid['objectiveValue']:.4f}")
```

### Visualize Parameter Landscape

Create a heatmap of the parameter space:

```python
# Extract results for 2D heatmap
grid_data = []
for result in results:
    params = result['parameters']
    grid_data.append({
        'ionFlux': params.get('ionFlux', optimal['ionFlux']),
        'etchantFlux': params.get('etchantFlux', optimal['etchantFlux']),
        'objective': result['objectiveValue']
    })

df = pd.DataFrame(grid_data)

# Create pivot table for heatmap
pivot = df.pivot_table(
    values='objective',
    index='etchantFlux',
    columns='ionFlux'
)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='Objective Value (lower = better)')
plt.xlabel('ionFlux')
plt.ylabel('etchantFlux')
plt.title('Parameter Space Landscape')

# Add optimal point marker
opt_ion_idx = 2  # Middle of grid (optimal value)
opt_etch_idx = 2
plt.plot(opt_ion_idx, opt_etch_idx, 'r*', markersize=20, label='Optimum')

plt.legend()
plt.savefig('parameter_landscape.png', dpi=150)
print("\nParameter landscape heatmap saved")
```

## Step 3: Specific Parameter Combinations

Test hand-picked parameter combinations:

```python
# Define specific combinations to test
specific_combos = [
    {
        "ionFlux": 40.0,
        "etchantFlux": 700.0,
        "ionEnergy": 85.0,
        "neutralStickP": 0.30
    },
    {
        "ionFlux": 45.0,
        "etchantFlux": 650.0,
        "ionEnergy": 90.0,
        "neutralStickP": 0.25
    },
    {
        "ionFlux": 50.0,
        "etchantFlux": 600.0,
        "ionEnergy": 95.0,
        "neutralStickP": 0.20
    }
]

# Set up paired evaluation
evaluator.setVariableValuesPaired(specific_combos)
evaluator.setDistanceMetric("CCH")

# Run evaluation
specific_results = evaluator.evaluateGrid(
    evaluationName="specificCombinations",
    saveVisualization=True  # Save these for detailed analysis
)

# Analyze
print("\nSpecific combinations results:")
for i, result in enumerate(specific_results):
    print(f"\nCombination {i+1}:")
    print(f"  Objective: {result['objectiveValue']:.4f}")
    for param, value in result['parameters'].items():
        print(f"  {param}: {value:.4f}")
```

!!! tip "When to Use Paired Evaluation"
    Use `setVariableValuesPaired()` when:
    - You have specific combinations from literature
    - You want to test parameter trade-offs
    - Grid search is too expensive (too many parameters)

## Step 4: Repeatability Testing

Assess simulation variance by running identical parameters multiple times:

```python
# Load best parameters (can be from CSV directly)
best_params = evaluator.loadBestFromProgressCSV("run1_initialCalibration")

print(f"\nBest parameters from CSV:")
for param, value in best_params.items():
    print(f"  {param}: {value:.4f}")

# Run 10 identical evaluations
evaluator.setConstantParametersWithRepeats(
    parameters=best_params,
    numRepeats=10
)

evaluator.setDistanceMetric("CCH")

# Execute
repeat_results = evaluator.evaluateGrid(
    evaluationName="repeatabilityTest",
    saveVisualization=False
)

print(f"\nRepeatability test complete: {len(repeat_results)} runs")
```

### Analyze Variance

```python
# Extract objective values
objectives = [r['objectiveValue'] for r in repeat_results]

# Compute statistics
mean_obj = np.mean(objectives)
std_obj = np.std(objectives)
cv = (std_obj / mean_obj) * 100  # Coefficient of variation

print(f"\nRepeatability Statistics:")
print(f"  Mean objective: {mean_obj:.6f}")
print(f"  Std deviation:  {std_obj:.6f}")
print(f"  Min:            {min(objectives):.6f}")
print(f"  Max:            {max(objectives):.6f}")
print(f"  Range:          {max(objectives) - min(objectives):.6f}")
print(f"  CV:             {cv:.2f}%")

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(objectives, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Objective Value')
plt.ylabel('Frequency')
plt.title('Distribution of Repeated Evaluations')
plt.axvline(mean_obj, color='r', linestyle='--', label=f'Mean: {mean_obj:.4f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), objectives, 'o-')
plt.xlabel('Run Number')
plt.ylabel('Objective Value')
plt.title('Repeatability Over Runs')
plt.axhline(mean_obj, color='r', linestyle='--', label='Mean')
plt.axhline(mean_obj + 2*std_obj, color='gray', linestyle=':', label='±2σ')
plt.axhline(mean_obj - 2*std_obj, color='gray', linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('repeatability_analysis.png', dpi=150)
print("\nRepeatability plots saved")
```

### Interpret Repeatability

**Good repeatability:**
- ✅ CV < 5% - Excellent
- ✅ CV < 10% - Good
- ⚠️ CV < 20% - Acceptable

**Poor repeatability:**
- ❌ CV > 20% - Investigate causes:
  - Random number seed not controlled
  - Numerical instabilities
  - Grid resolution too coarse
  - Physical instabilities in process

## Step 5: Working with Incomplete Runs

Load and analyze runs that didn't complete:

```python
# Create new evaluator
eval_incomplete = fit.CustomEvaluator(project)

# Load incomplete run (no final-results.json)
eval_incomplete.loadOptimizationRun("incomplete_run_name")

# Check if complete
if eval_incomplete.isCompleteRun:
    print("Run completed successfully")
else:
    print("⚠ Run is incomplete - using best parameters from CSV")

# Load best parameters from CSV
best_from_csv = eval_incomplete.loadBestFromProgressCSV("incomplete_run_name")

print(f"\nBest parameters extracted from incomplete run:")
for param, value in best_from_csv.items():
    print(f"  {param}: {value:.4f}")

# Can now use for further evaluation
eval_incomplete.setConstantParametersWithRepeats(best_from_csv, numRepeats=5)
eval_incomplete.setDistanceMetric("CCH")
results_from_incomplete = eval_incomplete.evaluateGrid("continue_from_incomplete")
```

!!! info "New in ViennaFit 2.0"
    Incomplete run support automatically:
    - Loads best parameters from `progressBest.csv`
    - Loads fixed parameters from `progressAll_metadata.json`
    - Loads process sequence from `{runName}-processSequence.py`
    - Sets `isCompleteRun = False` flag

## Step 6: Compare Multiple Runs

Compare results from different evaluations:

```python
# Load results from multiple evaluations
import glob
import json

evaluation_names = [
    "parameterGrid_ionAndEtchant",
    "specificCombinations",
    "repeatabilityTest"
]

comparison = []
for eval_name in evaluation_names:
    report_path = f"{project.projectPath}/customEvaluations/{eval_name}/grid_evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

        comparison.append({
            'Evaluation': eval_name,
            'Total Evals': report['metadata']['totalEvaluations'],
            'Best Objective': report['summary']['bestObjectiveValue'],
            'Mean Objective': report['summary']['objectiveValueRange']['mean']
        })

# Display comparison
df_comp = pd.DataFrame(comparison)
print("\nEvaluation Comparison:")
print(df_comp.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Best objective comparison
axes[0].bar(df_comp['Evaluation'], df_comp['Best Objective'])
axes[0].set_ylabel('Best Objective Value')
axes[0].set_title('Best Result by Evaluation')
axes[0].tick_params(axis='x', rotation=45)

# Total evaluations
axes[1].bar(df_comp['Evaluation'], df_comp['Total Evals'], color='orange')
axes[1].set_ylabel('Number of Evaluations')
axes[1].set_title('Computational Cost')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('evaluation_comparison.png', dpi=150)
print("\nComparison plots saved")
```

## Step 7: Export Results

Export results for external analysis:

```python
# Export grid results to CSV
grid_report_path = f"{project.projectPath}/customEvaluations/parameterGrid_ionAndEtchant"
grid_csv = pd.read_csv(f"{grid_report_path}/grid_results_summary.csv")

print("\nGrid results CSV columns:")
print(grid_csv.columns.tolist())

# Export selected data
export_data = grid_csv[['evaluationNumber', 'ionFlux', 'etchantFlux',
                        'objectiveValue', 'executionTime']]
export_data.to_csv('grid_results_export.csv', index=False)
print("Exported grid results to grid_results_export.csv")

# Export as Excel for easy sharing
with pd.ExcelWriter('custom_evaluations_summary.xlsx') as writer:
    # Grid results
    grid_csv.to_excel(writer, sheet_name='ParameterGrid', index=False)

    # Repeatability results
    repeat_df = pd.DataFrame({
        'Run': range(1, 11),
        'Objective': objectives
    })
    repeat_df.to_excel(writer, sheet_name='Repeatability', index=False)

    # Summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'CV (%)'],
        'Value': [mean_obj, std_obj, min(objectives), max(objectives), cv]
    })
    summary_df.to_excel(writer, sheet_name='Statistics', index=False)

print("Excel summary created: custom_evaluations_summary.xlsx")
```

## Complete Example Script

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup
vps.setDimension(2)
vls.setDimension(2)

# Load project and optimization results
project = fit.Project().load("./projects/etchCalibration")
evaluator = fit.CustomEvaluator(project)
evaluator.loadOptimizationRun("run1_initialCalibration")

# Get optimal parameters
optimal = evaluator.getOptimalParameters()

# 1. Parameter grid evaluation
evaluator.setVariableValues({
    "ionFlux": [optimal["ionFlux"] * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]],
    "etchantFlux": [optimal["etchantFlux"] * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
})
evaluator.setFixedParameters({
    "ionEnergy": optimal["ionEnergy"],
    "neutralStickP": optimal["neutralStickP"]
})
evaluator.setDistanceMetric("CCH")
grid_results = evaluator.evaluateGrid("parameterGrid")

# 2. Repeatability test
best_params = evaluator.loadBestFromProgressCSV("run1_initialCalibration")
evaluator.setConstantParametersWithRepeats(best_params, numRepeats=10)
repeat_results = evaluator.evaluateGrid("repeatability")

# 3. Analyze
objectives = [r['objectiveValue'] for r in repeat_results]
print(f"Mean: {np.mean(objectives):.6f}, Std: {np.std(objectives):.6f}")
```

## Key Takeaways

✅ **CustomEvaluator is powerful** - Explore beyond optimization
✅ **Grid search** - Systematic parameter space exploration
✅ **Paired values** - Test specific combinations
✅ **Repeatability testing** - Assess simulation variance
✅ **Incomplete runs** - ViennaFit 2.0+ handles gracefully

## Common Questions

??? question "How fine should my parameter grid be?"
    **Balance resolution vs cost:**
    - Start coarse (3-5 values per parameter)
    - Refine interesting regions
    - For 4 parameters at 5 values each = 625 evaluations!

??? question "What's acceptable variance in repeatability?"
    **Depends on application:**
    - Research: CV < 5% excellent
    - Production: CV < 1% may be needed
    - If high variance, check grid resolution and numerical settings

??? question "Can I use different metrics for custom evaluation?"
    **Yes!** You can explore with different metrics:
    ```python
    evaluator.setDistanceMetric("CA")  # Different from optimization
    ```
    This helps understand how metric choice affects results.

## Next Steps

**Continue learning:**
- [Tutorial 3: Sensitivity Analysis](tutorial-3-sensitivity-analysis.md) - Identify important parameters
- [Tutorial 4: Multi-Domain Optimization](tutorial-4-multi-domain.md) - Universal parameters

**Advanced topics:**
- Parameter screening before optimization
- Response surface modeling
- Multi-objective optimization (track multiple metrics)
