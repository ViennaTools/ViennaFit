# Sensitivity Analysis Examples

This folder contains examples for parameter sensitivity analysis using ViennaFit.

## Files

### localSensitivityStudy.py
- Local sensitivity analysis around optimal parameter values
- Evaluates parameter sensitivity one at a time
- Uses results from previous optimization runs

### globalSensitivityStudy.py
- Global sensitivity analysis using sampling methods
- Evaluates parameter interactions across entire parameter space
- Provides statistical measures of parameter importance

### parameterStudies.py
- Custom parameter studies using different approaches
- Single parameter sweeps
- Two-parameter interaction studies
- Distance metric comparisons

## Prerequisites

Before running sensitivity analysis examples:
1. Complete the setup examples (project initialization and domain creation)
2. Run the basic optimization example to generate optimization results
3. Ensure optimization run "run1" exists in the project

## Usage

All examples can be run independently after completing the prerequisites:

1. `localSensitivityStudy.py` - for focused local analysis
2. `globalSensitivityStudy.py` - for comprehensive global analysis  
3. `parameterStudies.py` - for custom parameter exploration

## Output

Results are saved in the project's sensitivity analysis directories:
- `localSensitivityStudies/` - for local sensitivity results
- `globalSensitivityStudies/` - for global sensitivity results
- `customEvaluations/` - for custom parameter study results

Each analysis includes visualization files and detailed statistical reports.