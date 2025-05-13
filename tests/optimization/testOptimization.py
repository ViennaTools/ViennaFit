import fit
import viennaps2d as vps
import viennals2d as vls

project = fit.Project("testOptimization")
project.initialize()
project.setMode("3D")

opt = fit.Optimization("testOptimization", project)

InitialDomain = vps.Domain()
opt.setInitialDomain(InitialDomain)

processSequence = fit.ProcessSequence()

opt.setProcessSequence(processSequence)


