# Installation

This guide will walk you through installing ViennaFit and its dependencies.

## Prerequisites

Before installing ViennaFit, ensure you have:

### Required
- **Python 3.10 or higher** - ViennaFit uses modern Python features like pattern matching and new type hints
- **ViennaPS 4.0.0+** - The process simulation backend (see below for installation)
- **Git** - For cloning the repository

### Recommended
- **Virtual environment** - Keep ViennaFit dependencies isolated from other projects
- **pip 21.0+** - Latest pip version for best dependency resolution

## Step 1: Install ViennaPS

ViennaFit requires ViennaPS 4.0.0 or higher.

!!! note "ViennaPS Installation"
    ViennaPS installation instructions can be found at the [ViennaPS repository](https://github.com/ViennaTools/ViennaPS).

    Make sure to install ViennaPS 4.0.0 or later. ViennaFit 2.0+ is **not compatible** with ViennaPS 3.x.

To verify ViennaPS installation:

```bash
python -c "import viennaps; print(viennaps.__version__)"
```

## Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment keeps ViennaFit's dependencies separate from your system Python:

```bash
# Create virtual environment
python -m venv viennafit-env

# Activate (Linux/macOS)
source viennafit-env/bin/activate

# Activate (Windows)
viennafit-env\Scripts\activate
```

## Step 3: Clone ViennaFit Repository

```bash
git clone https://github.com/ViennaTools/ViennaFit
cd ViennaFit
```

## Step 4: Install ViennaFit

Install ViennaFit and all required dependencies:

```bash
pip install .
```

This will automatically install:

- `dlib` 19.24.0 - Global optimization
- `nevergrad` 1.0.12+ - Alternative optimization algorithms
- `numpy` 1.26.4 - Array operations
- `cma` 3.2.2 - CMA-ES algorithm
- `SALib` 1.5.1+ - Sensitivity analysis
- `matplotlib` 3.5+ - Plotting
- `pandas` 1.5+ - Data handling

## Step 5: Verify Installation

Test that ViennaFit is correctly installed:

```python
import viennafit as fit
print(f"ViennaFit version: {fit.__version__}")

# Quick sanity check
import viennaps as vps
import viennals as vls
print("All imports successful!")
```

Expected output:
```
ViennaFit version: 2.0.0
All imports successful!
```

## Optional Dependencies

### Bayesian Optimization (Ax/BoTorch)

For advanced Bayesian optimization capabilities, install additional packages:

```bash
pip install botorch>=0.15.1 gpytorch>=1.14 ax-platform>=1.1.2
```

!!! warning "Large Install"
    These packages are quite large (~500MB) and have many dependencies. Only install if you need Bayesian optimization.

To verify Bayesian optimization is available:

```python
try:
    import botorch
    import ax
    print("Bayesian optimization available!")
except ImportError:
    print("Bayesian optimization not installed (optional)")
```

## Troubleshooting

### ViennaPS Not Found

**Error:**
```
ModuleNotFoundError: No module named 'viennaps'
```

**Solution:**
1. Ensure ViennaPS is installed: `pip list | grep viennaps`
2. Check version: `python -c "import viennaps; print(viennaps.__version__)"`
3. Verify you're using Python 3.10+: `python --version`
4. Try reinstalling ViennaPS

### Import Errors with Dimension

**Error:**
```
RuntimeError: Dimension not set
```

**Solution:**
Always set dimensions before using ViennaPS/ViennaLS:

```python
import viennaps as vps
import viennals as vls

vps.setDimension(2)  # or 3 for 3D
vls.setDimension(2)  # match ViennaPS dimension
```

### dlib Installation Fails

**Error:**
```
ERROR: Could not build wheels for dlib
```

**Solution (Linux):**
```bash
# Install build dependencies
sudo apt-get install cmake build-essential

# Try again
pip install dlib==19.24.0
```

**Solution (macOS):**
```bash
# Install Xcode command line tools
xcode-select --install

# Install cmake
brew install cmake

# Try again
pip install dlib==19.24.0
```

### Version Conflicts

**Error:**
```
ERROR: Cannot install viennafit because these package versions have conflicting dependencies
```

**Solution:**
1. Use a fresh virtual environment
2. Update pip: `pip install --upgrade pip`
3. Install ViennaFit in editable mode: `pip install -e .`

### Python Version Too Old

**Error:**
```
SyntaxError: invalid syntax (pattern matching not available)
```

**Solution:**
ViennaFit requires Python 3.10+. Update your Python:

```bash
# Check current version
python --version

# Install Python 3.10+ (Ubuntu/Debian)
sudo apt-get install python3.10 python3.10-venv

# Create virtualenv with specific version
python3.10 -m venv viennafit-env
```

## Development Installation

If you plan to contribute to ViennaFit or modify the source code:

```bash
# Clone repository
git clone https://github.com/ViennaTools/ViennaFit
cd ViennaFit

# Install in editable mode
pip install -e .

# Install development dependencies (optional)
pip install pytest black isort
```

Editable mode (`-e`) means changes to the source code take effect immediately without reinstalling.

## Updating ViennaFit

To update to the latest version:

```bash
cd ViennaFit
git pull origin main
pip install --upgrade .
```

## Uninstalling

To remove ViennaFit:

```bash
pip uninstall viennafit
```

To also remove the virtual environment:

```bash
# Deactivate first
deactivate

# Remove directory
rm -rf viennafit-env
```

## System Requirements

### Operating Systems
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+ (tested)
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10/11 (with WSL recommended)

### Hardware
- **CPU**: Modern multi-core processor recommended (optimization is CPU-intensive)
- **RAM**: Minimum 4GB, 8GB+ recommended for large simulations
- **Disk**: 2GB for ViennaFit and dependencies, additional space for results

### Python Environment
- **Python**: 3.10, 3.11, or 3.12
- **pip**: 21.0 or higher
- **virtualenv** or **venv**: Highly recommended

## Next Steps

Once installation is complete:

1. **[Quick Start](quick-start.md)** - Run your first optimization in 15 minutes
2. **[Core Concepts](concepts.md)** - Understand ViennaFit fundamentals
3. **[Tutorial 1](../tutorials/tutorial-1-basic-optimization.md)** - Detailed optimization walkthrough

## Getting Help

If you encounter issues not covered here:

- Check [GitHub Issues](https://github.com/ViennaTools/ViennaFit/issues) for similar problems
- Search [ViennaPS documentation](https://github.com/ViennaTools/ViennaPS) for ViennaPS-specific issues
- Open a [new issue](https://github.com/ViennaTools/ViennaFit/issues/new) with detailed error messages
