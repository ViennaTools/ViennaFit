# Implant Branch: SIMS, Implant, and Anneal Calibration

This document explains what the `implant` branch adds to ViennaFit, how the new
code is organized, how the workflows are intended to be used, and what was
validated.

The short version: this branch adds a new `viennafit.implant` package for
calibrating ion implantation and annealing models from SIMS profiles and sheet
resistance data. It supports a fast pure-Python Pearson IV fit, a slower
ViennaPS simulation-in-the-loop implant calibration, and a ViennaPS/ViennaCS
anneal calibration workflow.

## Motivation

Ion implantation calibration is not quite the same problem as the existing
ViennaFit level-set optimization workflows.

The existing ViennaFit API is centered around optimizing process parameters so
that simulated geometries match target geometries or annotations. SIMS implant
calibration starts from one-dimensional experimental concentration profiles
instead:

- pre-anneal SIMS gives the as-implanted dopant depth distribution,
- post-anneal SIMS gives the diffused dopant depth distribution,
- sheet resistance constrains the electrically active dopant fraction.

The branch therefore adds a focused implant-calibration layer next to the
existing ViennaFit optimization framework. It does not replace the existing
`Project` / `Optimization` workflows.

## What Is Added

The branch adds:

- a new package at `src/viennafit/implant/`,
- examples at `examples/implant-fit/`,
- tests at `tests/test_implant.py`,
- lazy top-level imports in `src/viennafit/__init__.py`,
- small dependency/test metadata updates in `pyproject.toml`.

The new package provides:

- SIMS CSV loading and normalization,
- Pearson IV and dual-Pearson IV profile evaluation,
- analytical fitting of Pearson IV parameters to SIMS data,
- ViennaPS domain/profile utilities for blanket and masked implant domains,
- ViennaPS simulation-in-the-loop implant calibration,
- ViennaPS/ViennaCS anneal calibration against post-anneal SIMS and optional
  sheet resistance,
- shared optimizer, metric, residual, and depth-convention helpers.

## Physical Coordinate Convention

The most important convention in this branch is the depth convention.

ViennaPS examples use a coordinate system where the wafer surface is at
`y = 0`, the silicon substrate lies below the surface at `y < 0`, and material
above the wafer, such as screen oxide, mask, or air, lies at `y > 0`.

SIMS data, however, is normally reported as a positive depth into the substrate.

The implant package therefore converts ViennaPS cell centers to SIMS depth with:

```python
depth_nm = surface_position - cell_center[depth_axis]
```

For the default two-dimensional ViennaPS geometry:

```python
surface_position = 0.0
depth_axis = 1
```

This means:

- a ViennaPS cell center at `y = -10 nm` maps to SIMS depth `10 nm`,
- a ViennaPS cell center at `y = 0 nm` maps to SIMS depth `0 nm`,
- a ViennaPS cell center at `y = +2 nm` maps to SIMS depth `-2 nm`.

By default, SIMS extraction and SIMS initialization operate only on the
substrate side. Cells above the surface are excluded or initialized to zero
unless the caller explicitly opts out of `substrate_only=True`.

This convention is implemented in `sims_depth_from_center()` in
`src/viennafit/implant/_common.py` and used by both:

- `extract_depth_profile()` in `simprofile.py`,
- `init_domain_from_sims()` in `annealfit.py`.

## Package Structure

The package intentionally keeps a small module split. Each module has a distinct
responsibility.

### `src/viennafit/implant/__init__.py`

This is the public API surface of the implant package.

It exports:

- `SimsProfile`,
- `PearsonIVFitter`,
- `FitResult`,
- `pearson_profile`,
- `dual_pearson_profile`,
- `pearson_unnorm`,
- `extract_depth_profile`,
- `build_blanket_substrate_domain`,
- `build_masked_substrate_domain`,
- `SimulationCalibrator`,
- `AnnealCalibrator`,
- `AnnealFitResult`,
- `init_domain_from_sims`.

The module docstring shows the three main workflows:

- analytical SIMS fit,
- implant simulation calibration,
- anneal calibration.

### `src/viennafit/implant/_common.py`

This is a private helper module.

It contains logic shared by analytical fitting, implant simulation calibration,
and anneal calibration:

- `sims_depth_from_center()`,
- log-space residual loss,
- log-space R2,
- log-space RMSE,
- optimizer backend selection,
- objective-call budget enforcement,
- initial population handling for warm starts.

The important design choice here is that `n_budget` means an approximate
objective-call budget. This matters because simulation-in-the-loop calibration
can be expensive. The SciPy backend wraps the objective in a budget guard so
extra calls requested internally by SciPy return a penalty instead of silently
running many more simulations than requested.

Supported optimizer names are:

- `scipy`,
- `dlib`,
- `nevergrad`,
- `cma`.

SciPy is the default and is the backend used by the tests. The other backends
are optional and depend on their corresponding packages being installed.

### `src/viennafit/implant/pearsoniv.py`

This module contains only Pearson IV profile math.

It provides:

- `pearson_unnorm()`,
- `pearson_profile()`,
- `dual_pearson_profile()`.

The fitter and calibrators call these functions, but the module itself does not
know anything about CSV files, ViennaPS domains, optimizers, or plotting.

The profile functions are written defensively so invalid parameter combinations
do not crash the optimizer. Non-finite values are handled by the callers through
penalized objectives.

### `src/viennafit/implant/simsfit.py`

This module implements the pure-Python analytical SIMS fitting workflow.

It contains:

- `SimsProfile`,
- `FitResult`,
- `PearsonIVFitter`.

`SimsProfile` loads depth/concentration pairs from CSV files and stores a
normalized profile. The expected CSV format is simple: a depth column in nm and
a concentration column. The examples use:

- `depth_nm`,
- `concentration`.

`PearsonIVFitter` can fit:

- a single Pearson IV profile,
- a dual Pearson IV profile with a head component and a tail component.

The analytical fit is fast and does not require ViennaPS. It is useful for:

- quickly checking whether a SIMS profile is reasonable,
- producing a first ViennaPS `config.txt` parameter block,
- generating an initial guess for the simulation-in-the-loop calibrator.

`FitResult` stores the fitted parameters, fit metrics, the optimizer name, and
the number of objective evaluations. It can also print a summary, plot the
result, and emit a ViennaPS-style config block.

### `src/viennafit/implant/simprofile.py`

This module bridges ViennaPS domains and one-dimensional SIMS profiles.

It contains:

- `extract_depth_profile()`,
- `build_blanket_substrate_domain()`,
- `build_masked_substrate_domain()`.

`extract_depth_profile()` reads scalar data from a ViennaPS cell set, maps cell
centers to positive SIMS depths, bins by grid depth, and aggregates values
laterally.

Supported aggregations are:

- `max`, useful for a representative centerline or blanket-style comparison,
- `sum`, useful when a lateral integral is desired.

`build_blanket_substrate_domain()` creates the default geometry used by the
SIMS examples. It contains:

- Si substrate from `y = -substrate_depth` to `y = 0`,
- optional screen oxide from `y = 0` to `y = oxide_thickness`,
- air above the screen oxide.

There is no mask in the blanket domain. This is the correct default for
blanket-wafer SIMS calibration data.

`build_masked_substrate_domain()` creates a patterned geometry with:

- Si substrate,
- screen oxide,
- a hard mask,
- a centered opening.

This builder is included for future calibration against patterned-device data,
such as lateral straggle under a mask edge or junction depth near a mask
opening. It is not used by the default SIMS examples.

ViennaPS is imported lazily in this module. Importing `viennafit.implant` does
not require `viennaps`; only actually building a domain does.

### `src/viennafit/implant/simcalibrator.py`

This module implements implant simulation calibration.

It contains:

- `SimulationCalibrator`.

The calibrator takes:

- a `SimsProfile`,
- a ViennaPS domain factory,
- the implant species,
- the implant dose,
- tilt and screen-oxide settings,
- a flag selecting single or dual Pearson IV fitting.

For each optimizer candidate, it:

1. builds a fresh ViennaPS domain,
2. creates the corresponding ViennaPS implant model,
3. runs the ViennaPS implant process,
4. extracts the simulated dopant depth profile,
5. interpolates it to the SIMS depth grid,
6. compares the result against the SIMS target in log space by default.

The simulation profile returned by `_run_simulation()` is normalized. Amplitude
is applied exactly once in the objective/final metrics through the fitted
`log_amplitude`. This avoids double-amplitude ambiguity.

The final simulation is validated. If it fails or produces no usable profile,
the calibrator raises a clear `RuntimeError` instead of returning misleading
fit metrics.

`initial_guess` is meaningful. A result from `PearsonIVFitter` can be passed to
warm-start the simulation calibration. For SciPy, the initialized population
contains the guess.

### `src/viennafit/implant/annealfit.py`

This module implements anneal calibration.

It contains:

- `init_domain_from_sims()`,
- `AnnealFitResult`,
- `AnnealCalibrator`.

`init_domain_from_sims()` writes a one-dimensional SIMS profile into a ViennaPS
cell set. It interpolates the concentration by SIMS depth and writes the field
`{species}_total`, for example `P_total`.

If the input concentration is normalized, it rescales the profile using the
provided implant dose. It converts from `cm^-3` to `nm^-3`, because ViennaPS
domains use nm as their length unit in these examples.

`AnnealCalibrator` starts from a measured pre-anneal profile and calibrates the
anneal model against a measured post-anneal profile. Optionally, it also fits
solid solubility parameters using a target sheet resistance.

The fitted parameters are:

- `annealD0`,
- `annealEa`,
- optionally `annealSolidSolubilityC0`,
- optionally `annealSolidSolubilityEa`.

The anneal simulation uses ViennaPS for diffusion and ViennaCS/ViennaPS sheet
resistance machinery for Rsh. ViennaFit does not duplicate the Masetti mobility
model.

Like the implant calibrator, failed final simulations produce clear errors
instead of silently returning invalid results.

## Top-Level Import Behavior

This branch changes `src/viennafit/__init__.py` so imports are lazier.

The goal is:

```python
import viennafit
import viennafit.implant
```

should work without immediately importing `viennaps` or running any ViennaPS
setup code.

ViennaPS is only required when the user actually executes ViennaPS-dependent
workflows, such as:

- `build_blanket_substrate_domain()`,
- `build_masked_substrate_domain()`,
- `SimulationCalibrator.fit()` with a ViennaPS domain factory,
- `AnnealCalibrator.fit()`.

This matters because the analytical SIMS fitter is pure Python and should be
usable on machines where ViennaPS is not installed.

## Example Directory

The branch adds `examples/implant-fit/`.

The directory contains three Python scripts and two CSV files:

```text
examples/implant-fit/
├── fitSIMS.py
├── calibrateSimulation.py
├── calibrateAnneal.py
├── sims_5keV_P_Si.csv
└── sims_5keV_P_Si_postanneal.csv
```

### `fitSIMS.py`

This is the minimal analytical SIMS fitting example.

It:

1. loads `sims_5keV_P_Si.csv`,
2. fits a single Pearson IV profile by default,
3. optionally fits a dual Pearson IV profile with `--dual`,
4. prints the result as a ViennaPS `config.txt` block,
5. optionally plots the fitted profile.

It does not require ViennaPS.

Typical commands:

```bash
python3 examples/implant-fit/fitSIMS.py
python3 examples/implant-fit/fitSIMS.py --dual
python3 examples/implant-fit/fitSIMS.py --no-plot --n-budget 5
```

Use this script when:

- you want a quick first fit,
- you want a simple config block,
- you want to verify that a SIMS CSV is readable,
- you do not need to run a ViennaPS simulation.

### `calibrateSimulation.py`

This is the two-step implant calibration example.

Step 1 is analytical:

1. load the SIMS profile,
2. fit a dual Pearson IV profile directly to SIMS,
3. use the result as a first estimate.

Step 2 is optional and simulation-based:

1. build a blanket ViennaPS domain,
2. run ViennaPS ion implantation for each optimizer candidate,
3. extract the simulated depth profile,
4. compare it to SIMS,
5. return simulation-calibrated Pearson IV parameters.

By default, the script runs only Step 1. Pass `--sim` to run Step 2.

Typical commands:

```bash
python3 examples/implant-fit/calibrateSimulation.py
python3 examples/implant-fit/calibrateSimulation.py --no-plot --n-analytical 5
python3 examples/implant-fit/calibrateSimulation.py --sim --no-plot --n-analytical 5 --n-budget 3
```

The default geometry is a blanket domain, which represents a bare-wafer implant.

Use this script when:

- you want the complete implant calibration workflow,
- you have ViennaPS bindings available,
- you want parameters self-consistent with the actual ViennaPS implant process,
- you want to use the analytical fit as a warm start for simulation calibration.

### `calibrateAnneal.py`

This is the anneal calibration example.

It:

1. loads pre-anneal SIMS from `sims_5keV_P_Si.csv`,
2. loads post-anneal SIMS from `sims_5keV_P_Si_postanneal.csv`,
3. initializes a ViennaPS domain from the pre-anneal SIMS profile,
4. runs ViennaPS anneal simulation,
5. compares the post-anneal profile against the target post-anneal SIMS,
6. optionally includes sheet resistance in the objective,
7. prints anneal config parameters.

Typical commands:

```bash
python3 examples/implant-fit/calibrateAnneal.py --no-sim
python3 examples/implant-fit/calibrateAnneal.py --no-plot --n-budget 3
python3 examples/implant-fit/calibrateAnneal.py --rsh 1200 --n-budget 200
```

`--no-sim` is a dry-run mode. It prints the config keys involved in anneal
calibration without requiring a simulation.

Use this script when:

- you want to calibrate anneal diffusion parameters,
- you have both pre-anneal and post-anneal SIMS,
- you also have sheet resistance data,
- you want to calibrate electrical activation through solid solubility.

### When to Use Which Example Script:

`fitSIMS.py` is the quick pure-Python implementation. It is short, direct, and
useful for users who only want an analytical fit.

`calibrateSimulation.py` is the full implant workflow. It overlaps with
`fitSIMS.py` in Step 1, but it demonstrates how the analytical fit is used
as a warm start for ViennaPS simulation calibration.

`calibrateAnneal.py` is a different physical workflow. It calibrates annealing,
not implantation, and uses pre/post SIMS plus optional sheet resistance.

The three files map cleanly to three user questions:

- "Can I fit my SIMS profile quickly?" -> `fitSIMS.py`
- "Can I calibrate ViennaPS implant parameters against SIMS?" ->
  `calibrateSimulation.py`
- "Can I calibrate anneal parameters and activation against post-anneal data?"
  -> `calibrateAnneal.py`

## Geometry Used by the SIMS Examples

The default SIMS calibration geometry is blanket, flat wafer.

Both `calibrateSimulation.py` and `calibrateAnneal.py` use:

```python
build_blanket_substrate_domain(...)
```

That domain contains:

- Si substrate,
- optional screen oxide,
- air/top space.

It does not contain:

- a hard mask,
- a mask opening,
- patterned topography.

This is deliberate. SIMS calibration data is normally measured on blanket
wafers, so using a masked geometry would mix geometry effects into what should
be a one-dimensional implant/anneal model calibration.

The branch still provides `build_masked_substrate_domain()` for later use. A
masked domain makes sense for patterned-device calibration, for example:

- lateral straggle near a mask edge,
- mask shadowing,
- junction depth under an opening,
- comparison against TEM or device-structure measurements.

It is not the default for SIMS.

## Optimizer Semantics

The public fitting methods keep the argument name `n_budget`.

In this branch, `n_budget` means an approximate maximum number of objective
evaluations, not a SciPy iteration count.

This distinction matters because:

- an analytical objective is cheap,
- a ViennaPS simulation objective is expensive,
- users need predictable simulation-call counts.

The shared SciPy backend uses:

- a budgeted objective wrapper,
- differential evolution with a population derived from the budget and
  dimension,
- no local refinement for simulation calibration unless explicitly allowed,
- an initialized population when an initial guess is supplied.

The simulation examples use deliberately tiny budgets in smoke tests, such as
`--n-budget 3`, only to verify that the workflow runs. Those tiny budgets are
not expected to produce high-quality calibrations.

## Fit Metrics

The branch uses log-space metrics by default because SIMS profiles span orders
of magnitude.

Shared metrics include:

- log-space residual loss,
- log-space R2,
- log-space RMSE.

The metrics are centralized in `_common.py` so analytical, implant simulation,
and anneal calibration use consistent definitions.

## Amplitude Handling

The implant simulation calibration separates profile shape from amplitude.

The simulation profile is normalized. The fitted `log_amplitude` is applied in
the objective/final comparison exactly once.

This avoids ambiguity where both the simulation output and the objective could
accidentally apply amplitude scaling.

## Failure Handling

Simulation-in-the-loop optimizers need to tolerate bad candidate parameters.

During optimization:

- invalid parameter sets return a large penalty,
- failed or empty simulation profiles return a large penalty.

For the final fitted result:

- a failed final implant simulation raises `RuntimeError`,
- a failed final anneal simulation raises `RuntimeError`.

This prevents returning fit metrics for a result that did not actually simulate.

## ViennaPS and ViennaCS Integration Notes

The analytical parts of `viennafit.implant` do not require ViennaPS.

The simulation and anneal parts require ViennaPS Python bindings. Anneal and
sheet resistance also rely on ViennaCS functionality exposed through ViennaPS.

During validation of this branch, the local ViennaPS installation needed a few
binding/install fixes outside ViennaFit:

- `install_ViennaPS.py` needed a `--viennacs-dir` option to use a local
  ViennaCS checkout,
- ViennaPS needed to account for the current ViennaCS `AnnealMode` values,
  which expose `Explicit` and `GaussSeidel`,
- ViennaPS needed `DenseCellSet.setScalarData` exposed in Python so SIMS
  initialization can overwrite cell-set scalar fields cleanly.

ViennaFit includes a fallback for cell-set scalar insertion so the examples can
work with bindings that expose lower-level mesh scalar insertion but not
`DenseCellSet.setScalarData`. The cleaner long-term behavior is for ViennaPS to
expose `setScalarData`, matching ViennaCS.

## Tests

The branch adds `tests/test_implant.py`.

The pure-Python tests cover:

- SIMS CSV loading,
- profile normalization,
- finite/non-negative Pearson profile behavior,
- ViennaPS-to-SIMS depth transform,
- substrate-only filtering,
- analytical single Pearson IV fitting with a tiny budget,
- analytical dual Pearson IV fitting with a tiny budget,
- shared metric helpers,
- optimizer budget accounting with a fake objective.

The ViennaPS-dependent tests are guarded with:

```python
pytest.importorskip("viennaps")
```

Those tests cover:

- blanket-domain profile extraction from a ViennaPS implant simulation,
- `init_domain_from_sims()` writing nonzero dopant concentration into Si cells
  below `y = 0`,
- one tiny-budget `SimulationCalibrator.fit()` run,
- one tiny-budget `AnnealCalibrator.fit()` run that produces `P_active` and
  can compute optional sheet resistance.

## Validation Commands

The following checks were run successfully without ViennaPS:

```bash
python3 -m compileall -q src/viennafit/implant examples/implant-fit
PYTHONPATH=src pytest -q tests/test_implant.py
PYTHONPATH=src python3 examples/implant-fit/fitSIMS.py --no-plot --n-budget 5
PYTHONPATH=src python3 examples/implant-fit/calibrateSimulation.py --no-plot --n-analytical 5
PYTHONPATH=src python3 examples/implant-fit/calibrateAnneal.py --no-sim
PYTHONPATH=src python3 -c "import viennafit; import viennafit.implant; print('implant import ok')"
```

With local ViennaPS and ViennaCS installed into the test virtual environment,
the guarded integration tests and tiny smoke examples were also run:

```bash
PYTHONPATH=src /path/to/venv/bin/python -m pytest -q tests/test_implant.py
PYTHONPATH=src /path/to/venv/bin/python examples/implant-fit/calibrateSimulation.py --sim --no-plot --n-analytical 5 --n-budget 3
PYTHONPATH=src /path/to/venv/bin/python examples/implant-fit/calibrateAnneal.py --no-plot --n-budget 3
```

The ViennaPS-backed test run passed all tests.

The anneal smoke test now checks that the computed sheet resistance is finite.
This depends on the ViennaCS/ViennaPS `SheetResistance` helper using the same
positive-into-substrate depth convention as SIMS:

```python
depth_nm = surface_position - cell_center[depth_axis]
```

With the local ViennaCS/ViennaPS binding fixes installed, a default anneal run
with `--n-budget 200` produced a finite and reasonable sheet resistance:

```text
Rsh_sim = 1267.0 ohm/sq  (target 1200.0, err +5.6%)
```

Tiny smoke-test budgets are still only wiring checks. Real calibration should
use a larger budget.

## Dependency Changes

The branch updates `pyproject.toml` to include:

- SciPy as a dependency for the default optimizer backend,
- a `test` optional dependency group with pytest.

SciPy is required because the new implant package uses SciPy differential
evolution as the default global optimizer.

## Intended User Workflows

### Workflow 1: Quick Analytical SIMS Fit

Use this when only SIMS data is available or when a first parameter estimate is
needed.

```python
from viennafit.implant import SimsProfile, PearsonIVFitter

profile = SimsProfile.from_csv("sims_5keV_P_Si.csv")
result = PearsonIVFitter(profile, dual=True).fit(n_budget=1200)
print(result.to_config_string())
```

Output parameters can be pasted into a ViennaPS config file.

### Workflow 2: Implant Simulation Calibration

Use this when ViennaPS is available and the fitted Pearson IV moments should be
self-consistent with the ViennaPS implant process.

```python
import functools
from viennafit.implant import (
    SimsProfile,
    PearsonIVFitter,
    SimulationCalibrator,
    build_blanket_substrate_domain,
)

profile = SimsProfile.from_csv("sims_5keV_P_Si.csv")
initial = PearsonIVFitter(profile, dual=True).fit(n_budget=1200)

domain_factory = functools.partial(
    build_blanket_substrate_domain,
    grid_delta=1.0,
    x_extent=50.0,
    top_space=10.0,
    substrate_depth=80.0,
    oxide_thickness=2.0,
)

calibrator = SimulationCalibrator(
    sims_profile=profile,
    domain_factory=domain_factory,
    species="P",
    dose_cm2=1e14,
    screen_thickness=2.0,
    dual=True,
)

result = calibrator.fit(initial_guess=initial, n_budget=200)
print(result.to_config_string())
```

### Workflow 3: Anneal Calibration

Use this when pre-anneal SIMS, post-anneal SIMS, and optionally sheet
resistance are available.

```python
import functools
from viennafit.implant import (
    SimsProfile,
    AnnealCalibrator,
    build_blanket_substrate_domain,
)

pre = SimsProfile.from_csv("sims_pre.csv")
post = SimsProfile.from_csv("sims_post.csv")

domain_factory = functools.partial(
    build_blanket_substrate_domain,
    grid_delta=1.0,
    x_extent=50.0,
    top_space=10.0,
    substrate_depth=80.0,
    oxide_thickness=2.0,
)

calibrator = AnnealCalibrator(
    pre_anneal_sims=pre,
    post_anneal_sims=post,
    rsh_target=1200.0,
    anneal_temp_C=1000.0,
    anneal_time_s=30.0,
    domain_factory=domain_factory,
    species="P",
    dose_cm2=1e14,
)

result = calibrator.fit(n_budget=200)
print(result.to_config_string())
```
