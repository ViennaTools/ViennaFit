import viennafit as fit
import viennaps as vps
import os

# Load the project
p1 = fit.Project()
projectToLoad = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "../projects/exampleProject",
)
p1.load(projectToLoad)

# Example 1: Simple parameter sweep using CustomEvaluator
print("=== Parameter Sweep Example ===")
evaluator = fit.CustomEvaluator(p1)

# Load optimization results from a previous run
evaluator.loadOptimizationRun("run1")
evaluator.setDistanceMetric("CA+CSF")

# Define a simple parameter sweep for one parameter
singleParamValues = {
    "neutralRate": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
}

evaluator.setVariableValues(singleParamValues)
results = evaluator.apply(
    evaluationName="singleParameterSweep", saveVisualization=False
)

print(f"Evaluated {len(results)} parameter combinations")
bestResult = evaluator.getBestResult()
if bestResult:
    print(f"Best objective value: {bestResult['objectiveValue']:.6f}")
    print(f"Best neutralRate: {bestResult['parameters']['neutralRate']:.2f}")

# Example 2: Two-parameter study
print("\n=== Two-Parameter Study Example ===")
evaluator2 = fit.CustomEvaluator(p1)
evaluator2.loadOptimizationRun("run1")
evaluator2.setDistanceMetric("CA+CSF")

# Study interaction between two parameters
twoParamValues = {
    "neutralRate": [20.0, 35.0, 50.0],
    "ionRate": [10.0, 20.0, 30.0],
}

evaluator2.setVariableValues(twoParamValues)
results2 = evaluator2.apply(evaluationName="twoParameterStudy", saveVisualization=False)

print(f"Evaluated {len(results2)} parameter combinations")
bestResult2 = evaluator2.getBestResult()
if bestResult2:
    print(f"Best objective value: {bestResult2['objectiveValue']:.6f}")
    print("Best parameter combination:")
    for param, value in bestResult2["parameters"].items():
        print(f"  {param}: {value:.2f}")


print(
    "\nParameter studies completed. Check the customEvaluations folder for detailed results."
)
