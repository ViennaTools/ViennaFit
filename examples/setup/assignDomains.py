import viennals as vls
import viennaps as vps

import viennafit as fit
import os

# Set dimensions for 2D mode
vps.setDimension(2)
vls.setDimension(2)


# In this example we create initial and target domains and assign them to a project.

# Load the project
p1 = fit.Project()
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects/exampleProject"))
p1.load(projectPath)

# Define the paths to the annotation files for the bottom and target domains
annotationBottom = os.path.join(scriptDir, "../data/regular-cropped-SiO2.png.dat")
annotationTarget = os.path.join(scriptDir, "../data/regular-cropped-Nitride.png.dat")

gridDelta = 5

# Read the points from the annotation files into meshes for the bottom and target domains
meshBottom = vls.Mesh()
extentBottom = fit.readPointsFromFile(
    annotationBottom, meshBottom, gridDelta, mode="2D"
)

meshTarget = vls.Mesh()
extentTarget = fit.readPointsFromFile(
    annotationTarget, meshTarget, gridDelta, mode="2D"
)

# Create LS domains for the bottom and target
domainBottom = vls.Domain(
    [10, 410, 200, -450],
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)
domainTarget = vls.Domain(
    [10, 410, 200, -450],
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

vls.FromSurfaceMesh(domainBottom, meshBottom).apply()
vls.FromSurfaceMesh(domainTarget, meshTarget).apply()


# Create the initial domain with the bottom domain as a material
domainInitial = vps.Domain(
    gridDelta=gridDelta,
    xExtent=extentBottom[1] - extentBottom[0],  # width of the domain
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY,
)
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.Si)

# Assign the initial and target domains to the project
# The domains along with visualization meshes are stored in project/domains
p1.setInitialDomain(domainInitial)
p1.setTargetLevelSet(domainTarget)
