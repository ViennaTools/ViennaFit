# Tutorial 4: Multi-Domain Optimization

Learn how to optimize process parameters across multiple geometries simultaneously to find universal parameter sets.

**Estimated time:** 35-45 minutes

## What You'll Learn

- Add multiple domains to a project
- Write multi-domain process sequences
- Run multi-domain optimization
- Interpret aggregated metrics
- Visualize per-domain results
- Find robust parameter sets

## Prerequisites

- Completed [Tutorial 1](tutorial-1-basic-optimization.md)
- Understanding of [Core Concepts - Multi-Domain](../getting-started/concepts.md#single-vs-multi-domain)
- Familiarity with ViennaPS domain creation

## Why Multi-Domain Optimization?

**Single-domain limitations:**
- Parameters optimized for one specific geometry
- May not work well for different feature sizes
- Overfitting to specific test structure

**Multi-domain benefits:**
- Find **universal** parameters that work across geometries
- More robust to geometry variations
- Better process window
- Closer to real manufacturing scenarios

**Use cases:**
- Calibrate across different feature sizes (dense vs isolated)
- Match multiple experimental samples
- Find process parameters robust to pattern density
- Optimize for both 2D and 3D structures

## Scenario

You have etch profiles from two different structures:
1. Dense trenches (narrow spacing)
2. Isolated trenches (wide spacing)

Goal: Find etch parameters that match **both** profiles simultaneously.

## Step 1: Project Setup

```python
import viennafit as fit
import viennaps as vps
import viennals as vls
import json
import os

# Setup
vps.setDimension(2)
vls.setDimension(2)

# Create project for multi-domain optimization
project = fit.Project("multiDomainEtch", "./projects")
project.initialize()

print(f"Multi-domain project created at: {project.projectPath}")
```

## Step 2: Create Multiple Initial Domains

Create two different initial geometries:

```python
gridDelta = 5.0

# === Domain 1: Dense Trenches ===
print("\nCreating Domain 1: Dense trenches")

dense_domain = vps.Domain(
    gridDelta=gridDelta,
    xExtent=600.0,
    yExtent=400.0,
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY
)

# Substrate
dense_substrate = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(dense_substrate, vls.lsBox([0, -200], [600, 50])).apply()

# Mask with narrow openings (100nm openings, 50nm spacing)
dense_mask = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(dense_mask, vls.lsBox([0, 50], [600, 100])).apply()

# Create openings
for x_pos in [75, 225, 375, 525]:  # 4 narrow openings
    vls.BooleanOperation(
        dense_mask,
        vls.lsBox([x_pos - 50, 50], [100, 50]),
        vls.BooleanOperationEnum.RELATIVE_COMPLEMENT
    ).apply()

dense_domain.insertNextLevelSetAsMaterial(dense_substrate, vps.Material.Si)
dense_domain.insertNextLevelSetAsMaterial(dense_mask, vps.Material.SiO2)

# Add to project with name "dense"
project.addInitialDomain("dense", dense_domain)

dense_domain.saveSurfaceMesh(
    os.path.join(project.projectPath, "domains/initialDomain/dense.vtp"),
    True
)

# === Domain 2: Isolated Trench ===
print("Creating Domain 2: Isolated trench")

isolated_domain = vps.Domain(
    gridDelta=gridDelta,
    xExtent=600.0,
    yExtent=400.0,
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY
)

# Substrate (same as dense)
isolated_substrate = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(isolated_substrate, vls.lsBox([0, -200], [600, 50])).apply()

# Mask with single wide opening (300nm opening, isolated)
isolated_mask = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)
vls.MakeGeometry(isolated_mask, vls.lsBox([0, 50], [600, 100])).apply()

# Single wide opening
vls.BooleanOperation(
    isolated_mask,
    vls.lsBox([150, 50], [300, 50]),
    vls.BooleanOperationEnum.RELATIVE_COMPLEMENT
).apply()

isolated_domain.insertNextLevelSetAsMaterial(isolated_substrate, vps.Material.Si)
isolated_domain.insertNextLevelSetAsMaterial(isolated_mask, vps.Material.SiO2)

# Add to project with name "isolated"
project.addInitialDomain("isolated", isolated_domain)

isolated_domain.saveSurfaceMesh(
    os.path.join(project.projectPath, "domains/initialDomain/isolated.vtp"),
    True
)

print("\nBoth initial domains added to project")
print(f"  - 'dense': {len(dense_domain.getLevelSets())} level sets")
print(f"  - 'isolated': {len(isolated_domain.getLevelSets())} level sets")
```

## Step 3: Create Multiple Target Domains

Create target profiles for each geometry:

```python
# === Target 1: Dense Trenches Result ===
print("\nCreating Target 1: Dense trench profiles")

dense_target = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)

# Etched trenches (simulating experimental result)
vls.MakeGeometry(dense_target, vls.lsBox([0, -200], [600, 50])).apply()

# Remove material for each trench
for x_pos in [75, 225, 375, 525]:
    vls.BooleanOperation(
        dense_target,
        vls.lsBox([x_pos - 40, -200], [80, 180]),  # Etched regions
        vls.BooleanOperationEnum.RELATIVE_COMPLEMENT
    ).apply()

project.addTargetLevelSet("dense", dense_target)

dense_target_mesh = vls.Mesh()
vls.ToSurfaceMesh(dense_target, dense_target_mesh).apply()
writer = vls.VTKWriter(dense_target_mesh)
writer.setFileName(
    os.path.join(project.projectPath, "domains/targetDomain/dense_target.vtp")
)
writer.apply()

# === Target 2: Isolated Trench Result ===
print("Creating Target 2: Isolated trench profile")

isolated_target = vls.Domain(
    [-300, 300, -200, 200],
    [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
     vls.BoundaryConditionEnum.INFINITE_BOUNDARY],
    gridDelta
)

vls.MakeGeometry(isolated_target, vls.lsBox([0, -200], [600, 50])).apply()

# Single wide etched trench
vls.BooleanOperation(
    isolated_target,
    vls.lsBox([150, -200], [300, 200]),  # Deeper/wider etch
    vls.BooleanOperationEnum.RELATIVE_COMPLEMENT
).apply()

project.addTargetLevelSet("isolated", isolated_target)

isolated_target_mesh = vls.Mesh()
vls.ToSurfaceMesh(isolated_target, isolated_target_mesh).apply()
writer = vls.VTKWriter(isolated_target_mesh)
writer.setFileName(
    os.path.join(project.projectPath, "domains/targetDomain/isolated_target.vtp")
)
writer.apply()

print("\nBoth target domains added to project")
print(f"Domain pairing:")
print(f"  'dense' initial ↔ 'dense' target")
print(f"  'isolated' initial ↔ 'isolated' target")
```

!!! info "Domain Pairing"
    Domains are paired by name. The process will simulate both:
    - "dense" initial → compare to "dense" target
    - "isolated" initial → compare to "isolated" target

    Distance metrics are **summed** across all domain pairs.

## Step 4: Multi-Domain Process Sequence

**Critical:** Multi-domain process sequences have different signature!

```python
def multiDomainEtchProcess(
    domains: dict[str, vps.Domain],      # Dictionary of domains
    params: dict[str, float]             # Parameters (same for all)
) -> dict[str, vls.Domain]:              # Dictionary of results
    """
    Multi-domain SF6/C4F8 etching process.

    Applies same parameters to all domains (dense and isolated).
    Demonstrates loading effect: isolated features etch differently than dense.

    Parameters:
    - ionFlux: Ion flux density
    - etchantFlux: Neutral etchant flux
    - ionEnergy: Ion mean energy
    - neutralStickP: Neutral sticking probability
    """
    results = {}

    # Process each domain with same parameters
    for domain_name, domain in domains.items():
        print(f"  Processing domain: {domain_name}")

        # Create model
        model = vps.MultiParticleProcess()

        # Add particles
        model.addIonParticle(
            sourcePower=params["ionFlux"],
            meanEnergy=params["ionEnergy"],
            label="ion"
        )

        sticking = {
            vps.Material.Si: params["neutralStickP"],
            vps.Material.SiO2: 0.01
        }
        model.addNeutralParticle(sticking=sticking, label="etchant")

        # Rate function (same for all domains)
        def rateFunction(fluxes, material):
            if material == vps.Material.Si:
                return (fluxes["ion"] * 0.5 +
                       fluxes["etchant"] * params["etchantFlux"] * 0.01)
            elif material == vps.Material.SiO2:
                return fluxes["ion"] * 0.05
            return 0.0

        model.setRateFunction(rateFunction)

        # Run process
        process = vps.Process()
        process.setDomain(domain)
        process.setProcessModel(model)
        process.setProcessDuration(1.0)

        rayTracing = vps.RayTracingParameters()
        rayTracing.raysPerPoint = 300
        process.setParameters(rayTracing)

        process.apply()

        # Store result
        results[domain_name] = domain.getLevelSets()[0]

    return results
```

!!! warning "Process Sequence Signature"
    **Multi-domain signature:**
    ```python
    def process(domains: dict[str, vps.Domain], params: dict) -> dict[str, vls.Domain]:
    ```

    **Single-domain signature:**
    ```python
    def process(domain: vps.Domain, params: dict) -> vls.Domain:
    ```

    ViennaFit auto-detects based on type annotations.

## Step 5: Configure Multi-Domain Optimization

```python
# Create optimization
opt = fit.Optimization(project)
opt.setProcessSequence(multiDomainEtchProcess)

# Multi-domain mode is auto-detected from:
# 1. Process sequence signature (dict[str, Domain])
# 2. Project having multiple domains

print(f"\nMulti-domain mode detected: {opt.isMultiDomainProcess}")

# Define parameters (same as single-domain)
opt.setParameterNames([
    "ionFlux",
    "etchantFlux",
    "ionEnergy",
    "neutralStickP"
])

opt.setVariableParameters({
    "ionFlux": (10.0, 100.0),
    "etchantFlux": (100.0, 1000.0),
    "ionEnergy": (20.0, 200.0),
    "neutralStickP": (0.01, 0.9)
})

# Distance metric is applied to each domain, then summed
opt.setDistanceMetrics(
    primaryMetric="CCH",
    additionalMetrics=["CA"]
)

opt.setName("run1_multiDomain")
opt.setNotes(
    "Multi-domain optimization across dense and isolated trenches. "
    "Finding universal parameters that work for both geometries."
)

print("\nMulti-domain optimization configured")
print(f"Parameters: {opt.getParameterNames()}")
print(f"Domains: {list(project.initialDomains.keys())}")
```

## Step 6: Run Multi-Domain Optimization

```python
print("\n" + "="*60)
print("STARTING MULTI-DOMAIN OPTIMIZATION")
print("="*60)
print("Note: Twice as many simulations per evaluation!")

opt.apply(
    numEvaluations=100,
    saveVisualization=True
)

print("\n" + "="*60)
print("MULTI-DOMAIN OPTIMIZATION COMPLETE")
print("="*60)
```

**Runtime:** 2× single-domain (processes both geometries per evaluation)

**Objective function:**
```
Total Objective = CCH("dense" sim vs target) + CCH("isolated" sim vs target)
```

## Step 7: Analyze Multi-Domain Results

```python
# Load results
results_dir = os.path.join(
    project.projectPath,
    "optimizationRuns",
    "run1_multiDomain"
)

with open(os.path.join(results_dir, "run1_multiDomain-final-results.json")) as f:
    results = json.load(f)

print("\n" + "="*60)
print("MULTI-DOMAIN OPTIMIZATION RESULTS")
print("="*60)

print(f"\nTotal Objective Value: {results['bestScore']:.4f}")
print("  (Sum across all domains)")

print(f"\nBest Parameters:")
for param, value in results['bestParameters'].items():
    print(f"  {param:20s}: {value:.4f}")

# Check per-domain results if available
progress = pd.read_csv(os.path.join(results_dir, "progressBest.csv"))

# If multi-domain metrics tracked separately
if 'dense_CCH_value' in progress.columns and 'isolated_CCH_value' in progress.columns:
    last_eval = progress.iloc[-1]
    print(f"\nPer-Domain Performance:")
    print(f"  Dense trenches:    {last_eval['dense_CCH_value']:.4f}")
    print(f"  Isolated trench:   {last_eval['isolated_CCH_value']:.4f}")
```

### Visualize Per-Domain Results

```python
# Load progress data
progress = pd.read_csv(os.path.join(results_dir, "progressBest.csv"))

# Plot convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(progress['evaluationNumber'], progress['objectiveValue'], 'b-', linewidth=2)
plt.xlabel('Evaluation Number')
plt.ylabel('Total Objective Value')
plt.title('Multi-Domain Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
if 'dense_CCH_value' in progress.columns:
    plt.plot(progress['evaluationNumber'], progress['dense_CCH_value'],
            'o-', label='Dense', alpha=0.7)
    plt.plot(progress['evaluationNumber'], progress['isolated_CCH_value'],
            's-', label='Isolated', alpha=0.7)
    plt.xlabel('Evaluation Number')
    plt.ylabel('Per-Domain Objective')
    plt.title('Individual Domain Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "multi_domain_convergence.png"), dpi=150)
print(f"\nConvergence plots saved")
```

## Step 8: Validate on Each Domain

Test best parameters separately on each domain:

```python
best_params = results['bestParameters']

print("\nValidating on each domain separately...")

# === Validate Dense ===
dense_val_domain = vps.Domain(project.initialDomains["dense"])
result_dense = multiDomainEtchProcess(
    {"dense": dense_val_domain},
    best_params
)

dense_val_mesh = vls.Mesh()
vls.ToSurfaceMesh(result_dense["dense"], dense_val_mesh).apply()
writer = vls.VTKWriter(dense_val_mesh)
writer.setFileName(os.path.join(results_dir, "validation_dense.vtp"))
writer.apply()

# === Validate Isolated ===
isolated_val_domain = vps.Domain(project.initialDomains["isolated"])
result_isolated = multiDomainEtchProcess(
    {"isolated": isolated_val_domain},
    best_params
)

isolated_val_mesh = vls.Mesh()
vls.ToSurfaceMesh(result_isolated["isolated"], isolated_val_mesh).apply()
writer = vls.VTKWriter(isolated_val_mesh)
writer.setFileName(os.path.join(results_dir, "validation_isolated.vtp"))
writer.apply()

print(f"\nValidation results saved:")
print(f"  Dense: {results_dir}/validation_dense.vtp")
print(f"  Isolated: {results_dir}/validation_isolated.vtp")
print("\nOpen in ParaView to compare with targets")
```

## Step 9: Compare with Single-Domain Results

Compare universal parameters with domain-specific optimizations:

```python
# If you have single-domain results for comparison
single_domain_results = {
    "dense": {
        "score": 25.3,  # From single-domain optimization
        "params": {"ionFlux": 45.2, "etchantFlux": 720.1}
    },
    "isolated": {
        "score": 22.1,
        "params": {"ionFlux": 41.8, "etchantFlux": 680.5}
    }
}

multi_total = results['bestScore']

print("\nSingle-Domain vs Multi-Domain Comparison:")
print(f"\nSingle-domain (optimized separately):")
print(f"  Dense score: {single_domain_results['dense']['score']:.2f}")
print(f"  Isolated score: {single_domain_results['isolated']['score']:.2f}")
print(f"  Total: {sum(r['score'] for r in single_domain_results.values()):.2f}")

print(f"\nMulti-domain (universal parameters):")
print(f"  Total score: {multi_total:.2f}")

print(f"\nTrade-off:")
if multi_total < sum(r['score'] for r in single_domain_results.values()):
    print(f"  ✓ Multi-domain found BETTER universal parameters!")
else:
    diff = multi_total - sum(r['score'] for r in single_domain_results.values())
    print(f"  Multi-domain score is {diff:.2f} higher (expected)")
    print(f"  But parameters work for BOTH geometries!")
```

## Complete Example Script

```python
import viennafit as fit
import viennaps as vps
import viennals as vls

# Setup
vps.setDimension(2)
vls.setDimension(2)

# Create project
project = fit.Project("multiDomainEtch", "./projects").initialize()

# Add multiple domains (code from steps 2-3)
# ... domain creation ...

# Multi-domain process sequence
def multiDomainEtchProcess(domains: dict[str, vps.Domain],
                          params: dict[str, float]) -> dict[str, vls.Domain]:
    results = {}
    for name, domain in domains.items():
        # Process each domain
        # ... (code from step 4) ...
        results[name] = result_levelset
    return results

# Configure and run optimization
opt = fit.Optimization(project)
opt.setProcessSequence(multiDomainEtchProcess)
opt.setParameterNames(["ionFlux", "etchantFlux", "ionEnergy", "neutralStickP"])
opt.setVariableParameters({
    "ionFlux": (10.0, 100.0),
    "etchantFlux": (100.0, 1000.0),
    "ionEnergy": (20.0, 200.0),
    "neutralStickP": (0.01, 0.9)
})
opt.setDistanceMetrics(primaryMetric="CCH")
opt.setName("run1_multiDomain")

# Run
opt.apply(numEvaluations=100, saveVisualization=True)
```

## Key Takeaways

✅ **Multi-domain finds universal parameters** - Work across geometries
✅ **Signature change required** - `dict[str, Domain]` → `dict[str, Domain]`
✅ **Metrics are summed** - Total objective across all domains
✅ **More robust** - Better process window than single-domain
✅ **Computational cost** - N×domains simulations per evaluation

## Common Questions

??? question "How many domains should I use?"
    **Guidelines:**
    - Start with 2-3 representative geometries
    - Too many → slow convergence, high cost
    - Too few → may miss important variations

    **Choose domains that represent:**
    - Different feature sizes (dense vs isolated)
    - Different aspect ratios
    - Critical process corners

??? question "Can I weight domains differently?"
    Currently, all domains weighted equally (sum). For custom weighting:
    ```python
    # In process sequence, return weighted objectives
    # This requires modifying framework
    ```

??? question "What if parameters don't work for all domains?"
    **Possible causes:**
    - Physics prevents universal solution (e.g., loading effects)
    - Targets inconsistent (different etch depths)
    - Need process adjustments per geometry

    **Solutions:**
    - Relax targets for some domains
    - Add geometry-specific parameters
    - Use process adaptation (e.g., time per geometry)

## Next Steps

**You've completed all tutorials!** 🎉

**What to do next:**
- Apply to your real process calibration problems
- Combine techniques (sensitivity + multi-domain)
- Explore advanced optimizers (Bayesian optimization)
- Contribute examples to ViennaFit repository

**Advanced Topics:**
- Time-dependent multi-domain (different durations per geometry)
- Hierarchical optimization (coarse then fine)
- Multi-objective optimization (multiple metrics)
- Uncertainty quantification across domains

**Get involved:**
- Share your results with the ViennaFit community
- Report issues on [GitHub](https://github.com/ViennaTools/ViennaFit/issues)
- Contribute improvements via pull requests
