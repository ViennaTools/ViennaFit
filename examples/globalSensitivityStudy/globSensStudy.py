import fit
import viennaps2d as vps
import os

# Create project and load data
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "projects/exampleProject",
)
p1.load(projectToLoad)

# Create global sensitivity study
gss = fit.GlobalSensitivityStudy("gss1", p1)

# Use the process sequence from a previous optimization run
processPath = os.path.abspath(
    os.path.join(p1.projectPath, "optimizationRuns", "run1", "run1-processSequence.py")
)

gss.loadProcessSequence(processPath)

# Parameters will be automatically detected from function signature
gss.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Set fixed parameters
gss.setFixedParameters(
    {"ionEnergy": 100.0, "neutralStickP": 0.03, "ionPowerCosine": 300.0}
)

# Set variable parameters with ranges
gss.setVariableParameters(
    {
        "neutralRate": (1.0, 70.0),
        "ionRate": (1.0, 40.0),
    }
)

gss.setDistanceMetric("CA+CSF")
gss.setSamplingOptions(numSamples=100, secondOrder=True)

# Run the sensitivity analysis
gss.apply(saveVisualization=False)
