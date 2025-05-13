import fit
# print all available classes and functions in fit
# import inspect
# print("Available classes and functions in fit:")
# for name, obj in inspect.getmembers(fit):
#     if inspect.isclass(obj) or inspect.isfunction(obj):
#         print(f"{name}: {obj}")

import viennaps2d as vps

project = fit.Project("DepositionOptimization")
project.initialize()

project.load("DepositionOptimization")

project.load("/home/kostal/Software/Fit/tests/project/DepositionOptimization")
