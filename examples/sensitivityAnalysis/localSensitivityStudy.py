import viennafit as fit
import viennaps2d as vps
import os

# Load the project
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../projects/exampleProject",
)
p1.load(projectToLoad)

lss1 = fit.LocalSensitivityStudy("lss1", p1)

# Use the process sequence from a previous optimization run
processPath = os.path.abspath(
    os.path.join(p1.projectPath, "optimizationRuns", "run1", "run1-processSequence.py")
)

lss1.loadProcessSequence(processPath)

# Parameters will be automatically detected from function signature
lss1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Set fixed parameters
lss1.setFixedParameters(
    {"ionEnergy": 100.0, "neutralStickP": 0.03, "ionPowerCosine": 300.0}
)

# Set variable parameters with ranges
lss1.setVariableParameters(
    {
        "neutralRate": (1.0, 40.0, 70.0),
        "ionRate": (1.0, 5.0, 40.0),
    }
)

lss1.setDistanceMetric("CA+CSF")

lss1.validate()

lss1.apply()
