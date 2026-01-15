import os

import viennafit as fit

# In case the optimization run was terminated before the convergence criteria were met,
# the finalizeOptimization function can be used to properly finalize the optimization.
# This will ensure that the best found parameters and corresponding domain are saved, plots
# are generated, and the optimization run is marked as completed.

p1 = fit.Project()
# Path to the example project created by setup scripts
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectToLoad = os.path.abspath(os.path.join(scriptDir, "../../projects/exampleProject"))
p1.finalizeOptimizationRun("run1")  # Specify the name of the optimization run to finalize
