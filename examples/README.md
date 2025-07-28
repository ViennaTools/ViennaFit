# ViennaFit Examples

This directory contains examples demonstrating the capabilities of the ViennaFit package

## Directory Structure

### data/
Contains example annotation data files used throughout the examples:
- `regular-cropped-SiO2.png.dat` - Bottom domain annotation
- `regular-cropped-Nitride.png.dat` - Target domain annotation

### setup/
Project initialization and domain creation examples:
- `initializeProject.py` - Create and initialize a ViennaFit project
- `createInitialDomain.py` - Set up initial and target domains from annotation data

### optimization/
Parameter optimization examples:
- `basicOptimization.py` - Basic process parameter optimization
- `customEvaluatorExample.py` - Custom evaluation after an optimization run

### sensitivityAnalysis/
Parameter sensitivity analysis examples:
- `localSensitivityStudy.py` - Local sensitivity analysis
- `globalSensitivityStudy.py` - Global sensitivity analysis with sampling
- `parameterStudies.py` - Custom parameter studies and comparisons

## Usage Workflow

### 1. Setup Phase
Run the setup examples in order:
```bash
cd setup/
python initializeProject.py
# Copy annotation files from data/ to project/domains/annotations/
python createInitialDomain.py
```

### 2. Optimization Phase
Run optimization examples:
```bash
cd optimization/
python basicOptimization.py
python customEvaluatorExample.py  # (after optimization completes)
```

### 3. Analysis Phase
Run sensitivity analysis examples (after optimization):
```bash
cd sensitivityAnalysis/
python localSensitivityStudy.py
python globalSensitivityStudy.py
python parameterStudies.py
```

## Key Concepts

- **Process Sequences**: Define the simulation workflow with parameters to optimize
- **Distance Metrics**: Methods for comparing simulation results with target domains
- **Parameter Optimization**: Finding optimal parameter values to match target results
- **Sensitivity Analysis**: Understanding how parameter changes affect the objective

## Output

Results are saved in the project directory structure:
- `optimizationRuns/` - Optimization results and process sequences
- `localSensitivityStudies/` - Local sensitivity analysis results
- `globalSensitivityStudies/` - Global sensitivity analysis results
- `customEvaluations/` - Custom parameter study results

