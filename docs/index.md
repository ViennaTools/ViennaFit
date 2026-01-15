# ViennaFit Documentation

Welcome to ViennaFit - a Python package for optimizing and calibrating ViennaPS process simulation models by fitting parameters to experimental data.

## What is ViennaFit?

ViennaFit automates the task of finding optimal process parameters that make your ViennaPS simulations match experimental results. Instead of manually tweaking parameters through trial and error, ViennaFit uses advanced optimization algorithms to systematically search the parameter space and find the best fit.

Whether you're calibrating an etch model to match measured profiles, exploring how different process conditions affect outcomes, or analyzing which parameters matter most, ViennaFit provides the tools you need.

## Key Features

<div class="grid cards" markdown>

-   :material-target: **Multiple Optimization Algorithms**

    ---

    Choose from dlib (global optimization), Nevergrad (evolutionary algorithms), or Ax/BoTorch (Bayesian optimization) depending on your needs.

-   :material-ruler: **8 Distance Metrics**

    ---

    Compare simulations to targets using CA (area), CCH (Chamfer), CSF (sparse field), CCD (critical dimensions), and more.

-   :material-chart-line: **Sensitivity Analysis**

    ---

    Identify which parameters matter most using local (one-at-a-time) or global (Sobol indices) sensitivity analysis.

-   :material-view-grid: **Multi-Domain Support**

    ---

    Optimize process parameters across multiple geometries simultaneously to find universal parameter sets.

-   :material-grid: **Custom Parameter Evaluation**

    ---

    Explore parameter space with grid search, test specific combinations, or run repeatability tests to assess variance.

-   :material-file-restore: **Incomplete Run Recovery**

    ---

    Load and analyze optimization runs that were stopped early or didn't complete successfully.

-   :material-chart-box: **Comprehensive Reporting**

    ---

    Automatic generation of convergence plots, parameter evolution tracking, and detailed CSV/JSON outputs.

-   :material-fast-forward: **Production Ready**

    ---

    Battle-tested in research and production environments with ViennaPS 4.0.0+ integration.

</div>

## Quick Example

Here's a complete optimization in just a few lines:

```python
import viennafit as fit
import viennaps as vps

# Create project
project = fit.Project("etchCalibration", "./projects").initialize()
project.setInitialDomain(initialDomain)
project.setTargetLevelSet(targetProfile)

# Define process sequence
def processSequence(domain: vps.Domain, params: dict[str, float]):
    # Your simulation using params
    return simulatedProfile

# Set up and run optimization
opt = fit.Optimization(project)
opt.setProcessSequence(processSequence)
opt.setVariableParameters({
    "etchRate": (10, 100),
    "stickingProb": (0.01, 0.9)
})
opt.setDistanceMetrics(primaryMetric="CCH")  # Chamfer distance
opt.apply(numEvaluations=100)

# Results saved automatically!
```

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast: **15-Minute Quickstart**

    ---

    Get your first optimization running in 15 minutes with our Quick Start guide.

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-page-variant: **Step-by-Step Tutorials**

    ---

    Learn ViennaFit through comprehensive, hands-on tutorials covering all major features.

    [:octicons-arrow-right-24: Tutorials](tutorials/tutorial-1-basic-optimization.md)

-   :material-download: **Installation Guide**

    ---

    Detailed installation instructions including prerequisites and troubleshooting.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open: **Core Concepts**

    ---

    Understand the fundamental concepts of projects, domains, metrics, and optimization.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

</div>

## Typical Workflows

ViennaFit supports various workflows depending on your goals:

### Initial Calibration
Start from scratch to calibrate your process model to experimental data:

1. Load experimental profile as target domain
2. Define initial geometry
3. Create process sequence with parameters to optimize
4. Run optimization with appropriate distance metric
5. Validate results and refine if needed

**Best for**: First-time calibration, new process models

### Parameter Exploration
After optimization, explore the parameter landscape:

1. Load optimization results
2. Set up parameter grid around optimal values
3. Evaluate systematic combinations
4. Visualize parameter relationships
5. Identify robust parameter regions

**Best for**: Understanding parameter sensitivity, finding robust solutions

### Sensitivity Analysis
Identify which parameters matter most:

1. Define parameter ranges
2. Run Sobol or local sensitivity analysis
3. Interpret sensitivity indices
4. Focus optimization on important parameters

**Best for**: Model reduction, experimental design, uncertainty quantification

### Multi-Domain Optimization
Find universal parameters that work across different geometries:

1. Add multiple initial and target domains
2. Write multi-domain process sequence
3. Run optimization (metrics automatically aggregated)
4. Validate on each domain

**Best for**: Process window optimization, robust parameter sets

## What's New in v2.0

ViennaFit 2.0 brings major improvements:

- **ViennaPS 4.0.0 Integration**: Updated for latest ViennaPS API
- **Incomplete Run Support**: Load and analyze runs that didn't complete
- **Repeatability Testing**: Convenient API for variance analysis
- **Fixed Multi-Domain Detection**: Single-domain is now the safe default
- **CSV Parameter Loading**: Direct loading from progressBest.csv
- **Better Documentation**: This site! Complete tutorials and guides

See the [CHANGELOG](https://github.com/ViennaTools/ViennaFit/blob/main/CHANGELOG.md) for full details.

## Example Use Cases

ViennaFit is used for:

- **Plasma Etching**: Calibrate ion and neutral flux parameters to match trench profiles
- **Deposition Processes**: Fit sticking coefficients and growth rates to experimental film profiles
- **Multi-Step Processes**: Optimize complex sequences of etching, deposition, and planarization
- **Process Window Analysis**: Find robust parameter sets that work across different geometries
- **Model Validation**: Quantify how well process models reproduce experimental observations

## Community and Support

- **GitHub Repository**: [ViennaTools/ViennaFit](https://github.com/ViennaTools/ViennaFit)
- **Report Issues**: [GitHub Issues](https://github.com/ViennaTools/ViennaFit/issues)
- **ViennaTools Ecosystem**: Part of the [ViennaTools](https://github.com/ViennaTools) suite

## Next Steps

Ready to get started? Here's what we recommend:

1. [Install ViennaFit](getting-started/installation.md) - Set up your environment
2. [Quick Start](getting-started/quick-start.md) - Run your first optimization in 15 minutes
3. [Tutorial 1](tutorials/tutorial-1-basic-optimization.md) - Deep dive into optimization workflow
4. [Core Concepts](getting-started/concepts.md) - Understand the fundamentals

Happy optimizing! 🚀
