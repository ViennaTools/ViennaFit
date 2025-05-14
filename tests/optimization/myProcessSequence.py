from fit import ProcessSequence
import viennaps2d as vps

class myProcessSequence(ProcessSequence):
    """Custom process sequence implementing a deposition process."""
    
    def apply(self, params):
        """
        Apply a deposition process to the initial domain.
        
        Args:
            params: Dictionary with process parameters
        """
        if self.initialDomain is None:
            raise RuntimeError("No initial domain has been set")
            
        model = vps.MultiParticleProcess()

        # Set the parameters for the neutral
        sticking = {vps.Material.Si: params.neutralStickP}
        model.addNeutralParticle(sticking, label="neutral")

        # Set the parameters for the ion
        model.addIonParticle(sourcePower=params.ionAngle, meanEnergy=params.ionEnergy, label="ion")

        # Set the rate function 
        def rateFunction(fluxes, material):
            if material == vps.Material.Si:
                return fluxes[0] * params.neutralRate + fluxes[1] * params.ionRate
            else:
                return 0.
            
        model.setRateFunction(rateFunction)    

        domain = vps.Domain()
        domain.deepCopy(self.initialDomain)

        process = vps.Process()
        process.setDomain(domain)
        process.setProcessModel(model)
        process.setProcessDuration(1.0)
        process.apply()
        
        # Store result
        self.resultLevelSet = domain.getLevelSets()[-1]
        self.applied = True
        
        return self
