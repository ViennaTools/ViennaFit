import viennafit as fit
import viennaps as vps
import json
import os

# Create project and load data
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../projects/exampleProject",
)
p1.load(projectToLoad)

# Create global sensitivity study
gss = fit.GlobalSensitivityStudy("gss1", p1)

# Use the process sequence and the optimum from a previous optimization run
runDir = os.path.abspath(os.path.join(p1.projectPath, "optimizationRuns", "run1"))
processPath = os.path.join(runDir, "run1-processSequence.py")
with open(os.path.join(runDir, "run1-final-results.json")) as f:
    optimum = json.load(f)["bestParameters"]

gss.loadProcessSequence(processPath)

# Declare all parameters used by the process sequence
gss.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Keep the remaining parameters fixed at their optimal values
gss.setFixedParameters(
    {name: optimum[name] for name in ["ionEnergy", "neutralStickP", "ionPowerCosine"]}
)

# Set variable parameters with ranges
gss.setVariableParameters(
    {
        "neutralRate": (1.0, 150.0),
        "ionRate": (0.1, 10.0),
    }
)

# Same distance metric as used in the optimization
gss.setDistanceMetric("CCH")

# First-order and total-order indices only: numSamples * (numParams + 2) = 128
# evaluations. Second-order indices (secondOrder=True) need at least 500 samples
# per variable parameter, i.e. 1000 * (2 * 2 + 2) = 6000 evaluations here.
gss.setSamplingOptions(numSamples=32, secondOrder=False)

# Run the sensitivity analysis
gss.apply(saveComparison=False)
