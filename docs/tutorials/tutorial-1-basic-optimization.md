# Tutorial 1: Basic Optimization

Set up a ViennaFit project from two annotated contours and fit a deposition model to them.

This tutorial walks through the scripts in `examples/1-setup/` and `examples/2-optimization/`. Every code block below is taken from those scripts, so you can read along and run them as they are.

**Estimated time:** 20-30 minutes, of which about 10-15 minutes is the optimization itself

## What You'll Learn

- Create a ViennaFit project
- Build the initial and target domains from annotation files
- Write a process sequence with parameters to fit
- Set variable and fixed parameters, metrics and the optimizer
- Run the optimization and find your way around the results

## Prerequisites

- ViennaFit installed (see [Installation](../getting-started/installation.md))
- A copy of the ViennaFit repository, for the `examples/` folder
- Basic familiarity with ViennaPS process models

## Scenario

A trench in SiO2 is filled by a deposition step. We have two contours, as you would get them from annotating a cross-section image before and after the step:

| File | Role | Shape |
|---|---|---|
| `examples/0-example-data/regular-cropped-SiO2.dat` | Initial geometry | Trench, 140 nm wide and 300 nm deep, with vertical walls |
| `examples/0-example-data/regular-cropped-Nitride.dat` | Target | The same trench after deposition: a thick film on the field, an overhang at the trench opening, a thinner film on the walls and at the bottom |

Each file holds one point per line as `x y` in nm, ordered along the contour.

The goal is to find the parameters of a simple two-particle deposition model (one neutral, one ion) that turn the initial trench into the target.

## Step 1: Create the Project

Script: `examples/1-setup/initializeProject.py`

```python
import viennafit as fit
import os
import shutil
import glob

# Specify a path to the projects directory relative to this script's location
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects"))

project = fit.Project("exampleProject", projectPath).initialize()

# Copy annotation files from example-data to the project's annotations folder
exampleDataDir = os.path.join(scriptDir, "../0-example-data")
annotationsDir = os.path.join(projectPath, "exampleProject", "domains", "annotations")

for datFile in glob.glob(os.path.join(exampleDataDir, "*.dat")):
    shutil.copy(datFile, annotationsDir)
    print(f"Copied {os.path.basename(datFile)} to project annotations folder")
```

Run it:

```bash
cd examples/1-setup
python initializeProject.py
```

`fit.Project(name, path).initialize()` creates the project folder, here `projects/exampleProject` in the repository root. The script then copies the two annotation files into it, so that the project is self-contained:

```
projects/exampleProject/
├── exampleProject-info.json
└── domains/
    ├── annotations/        <- the two .dat files
    ├── initialDomain/
    ├── targetDomain/
    └── optimalDomains/
```

Further folders (`optimizationRuns/`, `customEvaluations/`, `locSensStudies/`, `globSensStudies/`) are added when the first run of that kind is made.

!!! note
    The project stores absolute paths. If you move the project folder, run the setup scripts again.

## Step 2: Assign the Initial and Target Domains

Script: `examples/1-setup/assignDomains.py`

### Read the annotations

```python
import viennals as vls
import viennaps as vps

import viennafit as fit
import os

# Set dimensions for 2D mode
vps.setDimension(2)
vls.setDimension(2)

# Load the project
p1 = fit.Project()
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects/exampleProject"))
p1.load(projectPath)

annotationBottom = os.path.join(
    projectPath, "domains", "annotations", "regular-cropped-SiO2.dat"
)
annotationTarget = os.path.join(
    projectPath, "domains", "annotations", "regular-cropped-Nitride.dat"
)

# Grid resolution for level set representation
gridDelta = 3  # nm

meshBottom = vls.Mesh()
extentBottom = fit.readPointsFromFile(
    annotationBottom,
    meshBottom,
    gridDelta,
    mode="2D",  # 2D mode expects "x y" per line; 3D mode expects "x y z"
)

meshTarget = vls.Mesh()
extentTarget = fit.readPointsFromFile(
    annotationTarget, meshTarget, gridDelta, mode="2D"
)
```

`fit.readPointsFromFile` connects the points of a file into a polyline mesh and returns the extent `[minX, maxX, minY, maxY]` of the contour, rounded to `gridDelta`.

### Convert them to level sets

```python
domainBottom = vls.Domain(
    [10, 410, -450, 200],  # [minX, maxX, minY, maxY] in nm
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

domainTarget = vls.Domain(
    [10, 410, -450, 200],  # Same extent as bottom for consistency
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

vls.FromSurfaceMesh(domainBottom, meshBottom).apply()
vls.FromSurfaceMesh(domainTarget, meshTarget).apply()
```

Both level sets use the same bounds and the same `gridDelta`. The bounds are set by hand here: they are slightly narrower than the annotated contours (which run from x = 0 to 420), so the contours cross the reflective side boundaries cleanly.

### Hand them to the project

```python
domainInitial = vps.Domain(
    gridDelta=gridDelta,
    xExtent=extentBottom[1] - extentBottom[0],  # Width computed from annotation
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY,
)

# Insert the bottom geometry as the initial material (SiO2 substrate)
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.SiO2)

p1.setInitialDomain(domainInitial)  # ViennaPS initial domain
p1.setTargetLevelSet(domainTarget)  # ViennaLS target domain
```

The initial domain is a ViennaPS domain, because the process sequence will run on it. The target is a ViennaLS level set, because it is only compared against.

Run it:

```bash
python assignDomains.py
```

The project now contains both domains, together with surface meshes for inspection:

```
domains/initialDomain/exampleProject-initialDomain-surface.vtp
domains/targetDomain/exampleProject-targetDomain-surface.vtp
```

!!! tip "Check the domains before you optimize"
    Open both `-surface.vtp` files in ParaView and overlay them. A flipped contour or a wrong extent is much easier to spot here than in the result of a failed optimization.

## Step 3: Write the Process Sequence

Script: `examples/2-optimization/basicOptimization.py`

A process sequence is a function that takes a copy of the initial domain and a dictionary of parameters, runs the simulation, and returns the level set that is compared with the target.

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

    result = domain.getLevelSets()[-1]

    return result
```

The model has five parameters:

| Parameter | Meaning | Effect on the profile |
|---|---|---|
| `neutralStickP` | Sticking probability of the neutral | High values deposit near the opening and build the overhang; low values give a conformal film |
| `neutralRate` | Deposition rate per unit of neutral flux | Overall film thickness |
| `ionPowerCosine` | Exponent of the ion angular distribution | High values give a narrow, vertical ion beam |
| `ionRate` | Deposition rate per unit of ion flux | Film on surfaces the ions can see: the field and the trench bottom |
| `ionEnergy` | Mean ion energy | Kept fixed in this example |

The positive rate makes the surface grow, so the process is a deposition. `domain.getLevelSets()[-1]` returns the topmost level set, which is the surface after the process.

!!! info
    The function can also live in its own file and be loaded with `opt1.loadProcessSequence("path/to/sequence.py")`. ViennaFit stores a copy of the sequence in every run folder either way.

## Step 4: Configure the Optimization

```python
p1 = fit.Project()
p1.load(projectToLoad)

opt1 = fit.Optimization(p1)

opt1.setProcessSequence(processSequence1)

# Declare all parameters used by the process sequence
opt1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Set variable parameters with ranges
opt1.setVariableParameters(
    {
        "neutralStickP": (0.001, 0.9),
        "ionPowerCosine": (1.0, 900.0),
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)

# Set fixed parameters
opt1.setFixedParameters({"ionEnergy": 100.0})

# Set distance metrics: CCH (Chamfer) as primary, CA as additional metric.
opt1.setDistanceMetrics(
    primaryMetric="CCH",
    additionalMetrics=["CA"],
)

opt1.setOptimizer("cma")

opt1.setName("run1")

opt1.setNotes("Basic optimization of the multiparticle process for deposition.")
```

What each call does:

- **`setParameterNames`** lists every key the process sequence reads from `params`.
- **`setVariableParameters`** gives the parameters to fit, each with `(lowerBound, upperBound)`.
- **`setFixedParameters`** gives the rest a constant value. Every declared parameter has to be either variable or fixed.
- **`setDistanceMetrics`** chooses what is minimized. `CCH` is the Chamfer distance between the simulated and the target surface, in nm. `CA` (the area between the two) is recorded for every evaluation but does not drive the optimizer.
- **`setOptimizer("cma")`** selects the CMA-ES optimizer.
- **`setName`** sets the name of the run folder, and **`setNotes`** is saved into it as `notes.txt`.

!!! tip "Choosing bounds"
    Bounds should enclose every value you consider plausible, and it helps to know what a unit of each parameter does. In this model one unit of `ionRate` deposits roughly 10 nm, one unit of `neutralRate` roughly 1 nm on the field. That is why the two rates have such different ranges. If the optimum ends up sitting on a bound, widen that bound and run again.

## Step 5: Run the Optimization

```python
opt1.apply(numEvaluations=200, saveComparison=True)
```

```bash
cd ../2-optimization
python basicOptimization.py
```

For each evaluation ViennaFit copies the initial domain, runs the process sequence with the parameters proposed by the optimizer, compares the result with the target and records the distance. With `saveComparison=True` it also saves the surfaces used for the comparison whenever a new best result is found.

A single evaluation takes a few seconds, so 200 evaluations finish in about 10-15 minutes on a desktop machine.

!!! note "Running the example again"
    If a run called `run1` already exists, the new run is saved as `run1_1`, then `run1_2`, and so on. Nothing is overwritten.

## Step 6: Look at the Results

Everything belonging to the run is in `projects/exampleProject/optimizationRuns/run1/`:

| File | Content |
|---|---|
| `run1-final-results.json` | Best score, best parameters, bounds, optimizer and number of evaluations |
| `progressAll.csv` | Every evaluation: parameters, metric values and timings |
| `progressBest.csv` | Only the evaluations that improved on the best result so far |
| `progress/run1-NNN.vtp` | Simulated surface of each new best evaluation |
| `plots/run1-convergence-all.png`, `run1-convergence-best.png` | Objective value over the evaluations |
| `plots/run1-parameter-development.png`, `run1-parameter-positions.png` | How the parameters moved, and where they sit inside their bounds |
| `run1-processSequence.py` | Copy of the process sequence used for the run |
| `notes.txt` | The notes set with `setNotes` |

The best surface is also copied to `domains/optimalDomains/`.

`run1-final-results.json` looks like this (values from one run, yours will differ):

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
    },
    "optimizer": "cma",
    "numEvaluations": 200
}
```

Three things to check after every run:

1. **The fit itself.** Open `domains/optimalDomains/run1-NNN.vtp` together with the target surface in ParaView. A Chamfer distance of 2-3 nm on a 3 nm grid means the two surfaces lie within about one grid cell of each other.
2. **Convergence.** In `plots/run1-convergence-best.png` the curve should have flattened out. If it is still dropping at the end, run more evaluations.
3. **Bounds.** In `plots/run1-parameter-positions.png` no parameter should sit on a bound. A value on a bound means the optimizer wanted to go further.

!!! note "Results vary from run to run"
    The simulation uses Monte Carlo ray tracing and the optimizer is stochastic, so two runs do not give identical numbers. [Tutorial 2](tutorial-2-custom-evaluation.md) shows how to measure this noise.

## If a Run Was Stopped Early

Script: `examples/2-optimization/optional-finalizeOptimization.py`

If you interrupt a run, the progress files are there but the final results, the best domain and the plots are missing. They can be written afterwards:

```python
p1 = fit.Project()
p1.load(projectToLoad)
p1.finalizeOptimizationRun("run1")
```

This is only needed for interrupted runs. A run that finishes normally is finalized automatically.

## Key Takeaways

- A project holds the initial domain, the target and everything computed from them.
- Annotated contours become domains through `readPointsFromFile` and `vls.FromSurfaceMesh`.
- A process sequence is a plain function `(domain, params) -> level set`.
- Every parameter of the sequence is declared, and is either variable (with bounds) or fixed.
- After a run, check the fit, the convergence and the bounds.

## Next Steps

- [Tutorial 2: Custom Evaluation](tutorial-2-custom-evaluation.md) - explore the neighbourhood of the optimum and test repeatability
- [Tutorial 3: Sensitivity Analysis](tutorial-3-sensitivity-analysis.md) - find out which parameters matter
- [Core Concepts](../getting-started/concepts.md) - background on projects, metrics and optimizers
