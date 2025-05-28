# ViennaFit
A Python package for fitting ViennaPS simulation models to experimental data.

## Installing
Installing into a clean virtual environment is recommended. 
Does not work with older ViennaPS versions.

```bash
# Clone the repository
git clone https://github.com/kenyastyle/Fit.git
cd Fit

# Install the package and dependencies
pip install .
```

## Requirements (will be installed automatically)
- Python 3.10+
- ViennaLS 4.3.1+
- ViennaPS d155e3f and later
- NumPy 1.24.0+
- dlib 19.24.0+

## To do:
- Add postprocessing 
- Add further optimizers
- Custom range + presets for localSensStudy
- Objective function options
- 3D implementation
- Add custom objective function

<!-- ## Building
The project can be built using
```bash
cd Fit
pyproject-build

``` -->
