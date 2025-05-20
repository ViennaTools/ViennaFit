import viennals2d as vls
import viennaps2d as vps

import fit

p1 = fit.Project()
p1.load("/home/kostal/Software/Fit/tests/optimization/Project1")


readPathBottom = "Project1/domains/annotations/regular-cropped-SiO2.png.dat"
readPathTarget = "Project1/domains/annotations/regular-cropped-Nitride.png.dat"

gridDelta = 5

meshBottom = vls.Mesh()
extentBottom = fit.readPointsFromFile(readPathBottom, meshBottom, gridDelta, mode="2D")

meshTarget = vls.Mesh()
extentTarget = fit.readPointsFromFile(readPathTarget, meshTarget, gridDelta, mode="2D")

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

meshBottomLS = vls.Mesh()
meshTargetLS = vls.Mesh()
vls.ToMesh(domainBottom, meshBottomLS).apply()
vls.ToMesh(domainTarget, meshTargetLS).apply()

vls.VTKWriter(meshBottomLS, "meshBottomLS.vtp").apply()
vls.VTKWriter(meshTargetLS, "meshTargetLS.vtp").apply()

meshBottomSurface = vls.Mesh()
meshTargetSurface = vls.Mesh()
vls.ToSurfaceMesh(domainBottom, meshBottomSurface).apply()
vls.ToSurfaceMesh(domainTarget, meshTargetSurface).apply()
vls.VTKWriter(meshBottomSurface, "meshBottomSurface.vtp").apply()
vls.VTKWriter(meshTargetSurface, "meshTargetSurface.vtp").apply()

domainInitial = vps.Domain()
domainInitial.insertNextLevelSetAsMaterial(domainBottom, vps.Material.Si)

p1.setInitialDomain(domainInitial)
p1.setTargetLevelSet(domainTarget)
