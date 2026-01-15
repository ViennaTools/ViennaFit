"""
Example: Assigning Initial and Target Domains to a Project

This example demonstrates loading experimental data from .dat annotation files
and assigning initial and target domains to a ViennaFit project.

The .dat files contain space-separated x y coordinates representing
experimental surface contours (e.g., from annotated SEM images).
"""

import viennals as vls
import viennaps as vps

import viennafit as fit
import os

# Set dimensions for 2D mode
vps.setDimension(2)
vls.setDimension(2)

# Load the project
p1 = fit.Project()
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.abspath(os.path.join(scriptDir, "../../projects/exampleProject"))
p1.load(projectPath)

# ============================================================================
# LOAD EXPERIMENTAL DATA FROM ANNOTATION FILES
# ============================================================================

# Define the paths to the annotation files
# These .dat files contain x y coordinates (in nm) extracted from experimental images
annotationBottom = os.path.join(scriptDir, "../0-example-data/regular-cropped-SiO2.dat")
annotationTarget = os.path.join(scriptDir, "../0-example-data/regular-cropped-Nitride.dat")

# Grid resolution for level set representation
gridDelta = 3  # nm

# Read bottom domain points
# - Creates a mesh (polyline) from sequential points in the file
# - Returns extent [minX, maxX, minY, maxY] rounded to gridDelta
meshBottom = vls.Mesh()
extentBottom = fit.readPointsFromFile(
    annotationBottom,
    meshBottom,
    gridDelta,
    mode="2D"  # 2D mode expects "x y" per line; 3D mode expects "x y z"
)

# Read target domain points using the same process
meshTarget = vls.Mesh()
extentTarget = fit.readPointsFromFile(
    annotationTarget, meshTarget, gridDelta, mode="2D"
)

# ============================================================================
# CREATE LEVEL SET DOMAINS
# ============================================================================

# Create level set domain for bottom (initial substrate geometry)
# Note: Using custom extent [10, 410, 200, -450] instead of computed extentBottom
# This allows precise control over simulation domain boundaries
domainBottom = vls.Domain(
    [10, 410, 200, -450],  # [minX, maxX, minY, maxY] in nm
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,  # X: symmetric (half-domain)
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,    # Y: infinite depth
    ],
    gridDelta,
)

# Create level set domain for target (experimental result to match)
domainTarget = vls.Domain(
    [10, 410, 200, -450],  # Same extent as bottom for consistency
    [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ],
    gridDelta,
)

# Convert meshes (polylines) to level set representation
# This creates signed distance fields where:
#   - Negative values = inside material
#   - Positive values = outside material
#   - Zero = surface boundary
vls.FromSurfaceMesh(domainBottom, meshBottom).apply()
vls.FromSurfaceMesh(domainTarget, meshTarget).apply()

# ============================================================================
# CREATE INITIAL ViennaPS DOMAIN
# ============================================================================

# Create the initial domain for process simulation
domainInitial = vps.Domain(
    gridDelta=gridDelta,
    xExtent=extentBottom[1] - extentBottom[0],  # Width computed from annotation
    boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY,
)

# Insert the bottom geometry as the initial material (Silicon substrate)
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.SiO2)

# ============================================================================
# ASSIGN DOMAINS TO PROJECT
# ============================================================================

# Assign the initial and target domains to the project
# The domains along with visualization meshes are stored in project/domains
p1.setInitialDomain(domainInitial) # ViennaPS initial domain
p1.setTargetLevelSet(domainTarget) # ViennaLS target domain

# The project is now ready for optimization / sensitivity analysis
