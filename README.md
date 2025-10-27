# ViennaFit
A Python package for fitting ViennaPS simulation models to experimental data.

## Installing
Installing into a clean virtual environment is recommended. 
Does not work with older ViennaPS versions.

```bash
# Clone the repository
git clone https://github.com/ViennaTools/ViennaFit
cd ViennaFit

# Install the package and dependencies
pip install .
```

## Requirements (will be installed automatically)
- ViennaPS 3.5.1+
- dlib 19.24.0+
- nevergrad 1.0.12+
- NumPy 1.26.4
- cma 3.2.2
- SALib 1.5.1+

### Optional if you want to use Bayesian optimization 
- botorch 0.15.1
- gpytorch 1.14
- ax-platform 1.1.2

## Python version
- Python 3.10+
