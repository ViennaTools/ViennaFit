from .fitProject import Project
from .fitProcessSequence import ProcessSequence
from viennaps2d import Domain

class Optimization:
    def __init__(self, name: str, project: Project):
        self.name = name
        self.project = project
        self.initialDomain = None
        self.resultLevelSet = None
        self.applied = False
        self.parameters = None
        self.processSequence = None
        # if self.project.mode == "3D":
        #     import viennaps3d as vpsModule
        # else:
        #     import viennaps2d as vpsModule
        # self.vps = vpsModule

    def setProcessSequence(self, processSequence: ProcessSequence):
        """Set the process sequence to be optimized"""
        self.processSequence = processSequence
        return self
    
    def setInitialDomain(self, passedDomain: Domain):
        """Set the initial domain for the optimization"""
        self.initialDomain = passedDomain
        # Reset result since we have a new initial domain
        self.resultLevelSet = None
        self.applied = False
        return self                 

    def optimize(self):
        # Implement optimization logic here
        pass
