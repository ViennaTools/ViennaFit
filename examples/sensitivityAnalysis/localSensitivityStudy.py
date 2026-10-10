import viennafit as fit
import viennaps as vps
import json
import os

# Load the project
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../projects/exampleProject",
)
p1.load(projectToLoad)

lss1 = fit.LocalSensitivityStudy("lss1", p1)

# Use the process sequence and the optimum from a previous optimization run
runDir = os.path.abspath(os.path.join(p1.projectPath, "optimizationRuns", "run1"))
processPath = os.path.join(runDir, "run1-processSequence.py")
with open(os.path.join(runDir, "run1-final-results.json")) as f:
    optimum = json.load(f)["bestParameters"]

lss1.loadProcessSequence(processPath)

# Declare all parameters used by the process sequence
lss1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Keep the remaining parameters fixed at their optimal values
lss1.setFixedParameters(
    {name: optimum[name] for name in ["ionEnergy", "neutralStickP", "ionPowerCosine"]}
)

# Set variable parameters as (lowerBound, pointOfInterest, upperBound):
# each one is varied on its own around the optimum, within the optimization bounds
lss1.setParameterSensitivityRanges(
    {
        "neutralRate": (1.0, optimum["neutralRate"], 150.0),
        "ionRate": (0.1, optimum["ionRate"], 10.0),
    }
)

# Same distance metric as used in the optimization
lss1.setDistanceMetric("CCH")

lss1.validate()

lss1.apply()
