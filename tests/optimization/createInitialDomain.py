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

opt1 = fit.Optimization("run1", p1)


def processSequence1(domain: vps.Domain, params: dict[str, float]):
    """
    Process sequence for optimization.

    Args:
        domain: Initial ViennaPS domain
        params: Parameters for the process as dictionary

    Returns:
        resulting ViennLS domain which will be compared with the target
    """
    model = vps.MultiParticleProcess()

    # Set the parameters for the neutral
    sticking = {vps.Material.Si: params["neutralStickP"]}  # Use dict access
    model.addNeutralParticle(sticking, label="neutral")

    # Set the parameters for the ion
    model.addIonParticle(
        sourcePower=params["ionPowerCosine"],  # Use dict access
        meanEnergy=params["ionEnergy"],  # Use dict access
        label="ion",
    )

    def rateFunction(fluxes, material):
        if material == vps.Material.Si:
            return (
                fluxes[0] * params["neutralRate"] + fluxes[1] * params["ionRate"]
            )  # Use dict access
        return 0.0

    model.setRateFunction(rateFunction)

    process = vps.Process()
    process.setDomain(domain)
    process.setProcessModel(model)
    process.setProcessDuration(1.0)
    process.apply()

    result = domain.getLevelSets()[-1]

    return result


# Set up optimization
opt1.setProcessSequence(processSequence1)

# Parameters will be automatically detected from function signature
opt1.setParameterNames(
    ["neutralStickP", "ionPowerCosine", "neutralRate", "ionRate", "ionEnergy"]
)

# Set variable parameters with ranges
opt1.setVariableParameters(
    {
        "neutralStickP": (0.001, 0.9),
        "ionPowerCosine": (1.0, 900.0),
        "neutralRate": (1.0, 70.0),
        "ionRate": (1.0, 40.0),
    }
)

# Set fixed parameters
opt1.setFixedParameters({"ionEnergy": 100.0})

opt1.setDistanceMetric("CA+CSF")

# Validate before optimization
opt1.validate()

opt1.apply()
