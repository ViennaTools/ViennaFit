import viennaps2d as vps
import viennals2d as vls
from typing import Optional, Any, Dict


class ProcessSequence:
    """
    Parent class for a process sequence. The user should implement the apply() method in a subclass.

    """

    def __init__(self):
        """Initialize a ProcessSequence object."""
        self.initialDomain = None
        self.resultLevelSet = None
        self.applied = False

    def setInitialDomain(self, domain):
        """
        Set the initial domain for the process sequence.

        Args:
            domain: A ViennaPS domain object representing the initial state

        Returns:
            self: For method chaining
        """
        self.initialDomain = domain
        # Reset result since we have a new initial domain
        self.resultLevelSet = None
        self.applied = False
        return self

    def apply(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Apply the process sequence to the initial domain.
        This method should be implemented by the user.

        Args:
            parameters: Optional dictionary of process parameters

        Returns:
            self: For method chaining
        """
        raise NotImplementedError("The 'apply' method must be implemented by the user.")

    def getResultLevelSet(self):
        """
        Get the resulting level set after applying the process sequence.

        Returns:
            A ViennaLS domain with the level set which will be used for comparison
            with the experimental data.

        Raises:
            RuntimeError: If apply() hasn't been called yet or if no initial domain was set
        """
        if self.initialDomain is None:
            raise RuntimeError(
                "No initial domain has been set. Call setInitialDomain() first."
            )

        if not self.applied:
            raise RuntimeError("Process has not been applied yet. Call apply() first.")

        return self.resultLevelSet
