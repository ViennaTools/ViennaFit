import viennafit as fit

# # Option 1: Create and initialize in separate steps
# project = fit.Project("exampleProject")
# project.initialize()

# # Option 2: Chain methods
# project = fit.Project("exampleProject").initialize()

# Option 3: Specify custom directory - one above the current directory
project = fit.Project("exampleProject", "../../projects").initialize()

# It is recommended to then copy the annotations from the exampleData folder into
# the exampleProject/domains/annotations directory.
