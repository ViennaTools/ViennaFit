import viennafit as fit
import os

# Specify a path to the projects directory relative to this script's location
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects"))

project = fit.Project("exampleProject", projectPath).initialize()

# It is recommended to then copy the annotations from the data folder into
# the exampleProject/domains/annotations directory.
