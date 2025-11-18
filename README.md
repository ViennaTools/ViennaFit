# ViennaFit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python package for parameter optimization and calibration of ViennaPS process simulation models. ViennaFit automates the fitting of simulation parameters to match experimental data through various optimization algorithms and distance metrics.

## Quick Example

```python
import viennafit as fit
import viennaps as vps

# Create and initialize project
project = fit.Project("myOptimization", "./projects").initialize()
project.setInitialDomain(initialDomain)
project.setTargetLevelSet(targetLevelSet)

# Set up optimization
opt = fit.Optimization(project)
opt.setProcessSequence(processSequence)
opt.setVariableParameters({"param1": (0.1, 1.0), "param2": (10, 100)})
opt.setDistanceMetrics(primaryMetric="CCH")  # Chamfer distance

# Run optimization
opt.setName("run1")
opt.apply(numEvaluations=100)
```

## Key Features

- **Multiple Optimization Algorithms**: dlib, Nevergrad, Ax/BoTorch (Bayesian optimization)
- **8 Distance Metrics**: Compare Area (CA), Chamfer (CCH), Sparse Field (CSF), Critical Dimensions (CCD), and more
- **Sensitivity Analysis**: Both local and global (Sobol) methods
- **Multi-Domain Support**: Optimize across multiple geometries simultaneously
- **Custom Parameter Evaluation**: Grid search and specific parameter combinations
- **Incomplete Run Recovery**: Load and analyze interrupted optimization runs
- **Comprehensive Reporting**: Convergence plots, parameter evolution, CSV/JSON outputs

## Installation

Installing into a clean virtual environment is recommended.

**Note:** ViennaFit 2.0.0+ requires ViennaPS 4.0.0+. For ViennaPS 3.5.1, use ViennaFit 1.x.

```bash
# Clone the repository
git clone https://github.com/ViennaTools/ViennaFit
cd ViennaFit

# Install the package and dependencies
pip install .
```

### Optional Dependencies

For Bayesian optimization (Ax/BoTorch):
```bash
pip install botorch>=0.15.1 gpytorch>=1.14 ax-platform>=1.1.2
```

## Requirements

**Python:** 3.10 or higher

**Dependencies** (installed automatically):
- ViennaPS >= 4.0.0
- dlib == 19.24.0
- nevergrad >= 1.0.12
- NumPy == 1.26.4
- cma == 3.2.2
- SALib >= 1.5.1
- matplotlib >= 3.5
- pandas >= 1.5

## Documentation

📚 **Full Documentation**: [https://viennatools.github.io/ViennaFit/](https://viennatools.github.io/ViennaFit/)

- [Quick Start Guide](https://viennatools.github.io/ViennaFit/getting-started/quick-start/) - Get running in 15 minutes
- [Tutorials](https://viennatools.github.io/ViennaFit/tutorials/tutorial-1-basic-optimization/) - Step-by-step walkthroughs
- [Examples](examples/) - Working code examples

## Quick Start

1. **Create a project**:
   ```python
   project = fit.Project("myProject", "./projects").initialize()
   ```

2. **Assign domains** (initial geometry and target to match):
   ```python
   project.setInitialDomain(initialDomain)
   project.setTargetLevelSet(targetLevelSet)
   ```

3. **Define your process** in a function that takes parameters:
   ```python
   def processSequence(domain: vps.Domain, params: dict[str, float]):
       # Your simulation code using params
       return resultLevelSet
   ```

4. **Run optimization**:
   ```python
   opt = fit.Optimization(project)
   opt.setProcessSequence(processSequence)
   opt.setVariableParameters({"param": (min, max)})
   opt.setDistanceMetrics(primaryMetric="CCH")
   opt.apply(numEvaluations=100)
   ```

## Project Structure

After initialization, ViennaFit creates:
```
projectName/
├── domains/
│   ├── initialDomain/      # Your starting geometry
│   ├── targetDomain/       # Goal to match
│   └── annotations/        # Measurement data
├── optimizationRuns/       # Results from optimizations
├── customEvaluations/      # Custom parameter evaluations
├── localSensitivityStudies/
└── globalSensitivityStudies/
```

## Citation

If you use ViennaFit in your research, please cite:

```bibtex
@software{viennafit,
  title = {ViennaFit},
  author = {Roman Kostal},
  year = {2025},
  url = {https://github.com/ViennaTools/ViennaFit}
}
```

## License

ViennaFit is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: https://viennatools.github.io/ViennaFit/
- **Issues**: https://github.com/ViennaTools/ViennaFit/issues
- **Examples**: See [examples/](examples/) directory
