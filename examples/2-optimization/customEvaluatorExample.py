import viennafit as fit
import os

# Load the project
p1 = fit.Project()
# Path to the example project created by setup scripts
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectToLoad = os.path.abspath(os.path.join(scriptDir, "../../projects/exampleProject"))
p1.load(projectToLoad)

# Create a custom evaluator instance
evaluator = fit.CustomEvaluator(p1)

# Load optimization results from a previous run
# This assumes you have already run the basicOptimization.py example
evaluator.loadOptimizationRun("run1")

# Set distance metric (same as used in optimization)
evaluator.setDistanceMetric("CCH")
evaluator.setAdditionalMetrics(["CSF"])

# Define parameter grids to evaluate
# This will vary neutralRate and ionRate around their optimal values
variableValues = {
    "neutralRate": [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    "ionRate": [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
}

evaluator.setVariableValues(variableValues)

# Print the optimal parameters for reference
optimalParams = evaluator.getOptimalParameters()
print("Optimal parameters from optimization run:")
for paramName, value in optimalParams.items():
    print(f"  {paramName}: {value}")

# Run the grid evaluation
# This will evaluate all combinations of the variable parameters
# while keeping other parameters at their optimal values
results = evaluator.apply(
    evaluationName="parameterSweep1", saveVisualization=True
)

# Get and display the best result from the grid
bestResult = evaluator.getBestResult()
if bestResult:
    print(f"\nBest result from grid evaluation:")
    print(f"  Objective value: {bestResult['objectiveValue']:.6f}")
    print(f"  Parameters:")
    for paramName, value in bestResult["parameters"].items():
        print(f"    {paramName}: {value:.6f}")
else:
    print("No valid results found in grid evaluation.")

# Save a detailed report
evaluator.saveGridReport()
