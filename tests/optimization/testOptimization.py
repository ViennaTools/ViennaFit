import fit
import viennaps2d as vps
import viennals2d as vls

project = fit.Project("Project1")
project.initialize()
project.setMode("2D")

opt = fit.Optimization("run1", project)

InitialDomain = vps.Domain()
opt.setInitialDomain(InitialDomain)

processSequence = fit.ProcessSequence()

opt.setProcessSequence(processSequence)

# Step 1: Define all available parameters
opt.addParameters([
    "neutralStickP", 
    "ionAngle", 
    "ionEnergy", 
    "neutralRate", 
    "ionRate"
])

# Step 2: Set fixed parameters (these won't change during optimization)
opt.setFixedParameters({
    "neutralStickP": 0.8,
    "ionAngle": 0.0
})

# Step 3: Set variable parameters with their ranges
# Format: parameterName: (minValue, maxValue)
opt.setVariableParameters({
    "ionEnergy": (50.0, 500.0),
    "neutralRate": (0.1, 2.0),
    "ionRate": (0.05, 1.0)
})

# Display current parameter values
params = opt.getParameterDict()
print("Parameter values before optimization:")
for name, value in params.items():
    print(f"  {name}: {value}")

# Save the parameter configuration
opt.saveParameters()

# Run the optimization
# opt.optimize()  # Uncomment once you've implemented the actual optimization logic

print("\nOptimization framework configured successfully!")
