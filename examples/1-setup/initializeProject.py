import viennafit as fit
import os
import shutil
import glob

# Specify a path to the projects directory relative to this script's location
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects"))

project = fit.Project("exampleProject", projectPath).initialize()

# Copy annotation files from example-data to the project's annotations folder
exampleDataDir = os.path.join(scriptDir, "../0-example-data")
annotationsDir = os.path.join(projectPath, "exampleProject", "domains", "annotations")

for datFile in glob.glob(os.path.join(exampleDataDir, "*.dat")):
    shutil.copy(datFile, annotationsDir)
    print(f"Copied {os.path.basename(datFile)} to project annotations folder")
