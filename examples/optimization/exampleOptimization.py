import viennaps2d as vps
import os

import viennafit as fit

p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "projects/exampleProject",
)
p1.load(projectToLoad)

opt1 = fit.Optimization("run1", p1)


# Define a process sequence function whose parameters will be optimized.
# This function can also be specified in a separate file and imported using
# opt1.loadProcessSequence("path/to/sequence.py")
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

opt1.apply(saveVisualization=True)
