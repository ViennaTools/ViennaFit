import viennafit as fit
import os

# Load the project
p1 = fit.Project()
# Path to the example project created by setup scripts
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectToLoad = os.path.abspath(
    os.path.join(scriptDir, "../../projects/exampleProject")
)
p1.load(projectToLoad)

# Create a custom evaluator instance
evaluator = fit.CustomEvaluator(p1)

# Load optimization results from a previous run
# This assumes you have already run the basicOptimization.py example
evaluator.loadOptimizationRun("run1")

# Get the optimal parameters from the optimization run
bestParams = evaluator.getOptimalParameters()

print("Optimal parameters from optimization run:")
for paramName, value in bestParams.items():
    print(f"  {paramName}: {value}")

# Set up repeated evaluations with the optimal parameters
# This runs the same parameters multiple times to assess reproducibility
evaluator.setConstantParametersWithRepeats(bestParams, numRepeats=10)

# Set distance metric (same as used in optimization)
evaluator.setDistanceMetric("CCH")

# Run the repeated evaluation
# Uses the same process sequence as the loaded optimization run
results = evaluator.apply(evaluationName="repeatedEvaluation", saveVisualization=True)
