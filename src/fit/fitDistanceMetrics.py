from typing import Callable
import viennals2d as vls


class DistanceMetric:
    """Factory class for distance metrics used in optimization."""

    AVAILABLE_METRICS = ["CA", "CSF", "CNB", "CA+CSF", "CA+CNB"]

    @staticmethod
    def create(
        metricName: str,
    ) -> Callable[[vls.Domain, vls.Domain, bool], float]:
        """
        Create a distance metric function based on metric name.

        Args:
            metricName: Name of the distance metric to use

        Returns:
            Callable that takes (domain1, domain2, save_visualization) and returns
            (distance_value, (visualization_mesh1, visualization_mesh2))
        """
        if metricName == "CA+CSF":
            return DistanceMetric._compareAreaAndSparseField
        elif metricName == "CA":
            return DistanceMetric._compareArea
        elif metricName == "CSF":
            return DistanceMetric._compareSparseField
        else:
            raise ValueError(
                f"Invalid distance metric: {metricName}. "
                f"Available options are: {', '.join(DistanceMetric.AVAILABLE_METRICS)}"
            )

    @staticmethod
    def _compareArea(
        domain1: vls.Domain, domain2: vls.Domain, saveVisualization: bool = False
    ) -> float:
        """Compare domains using area mismatch."""
        ca = vls.CompareArea(domain1, domain2)
        if saveVisualization:
            mesh = vls.Mesh()
            ca.setOutputMesh(mesh)
        ca.apply()
        return ca.getAreaMismatch()

    @staticmethod
    def _compareSparseField(
        domain1: vls.Domain, domain2: vls.Domain, saveVisualization: bool = False
    ) -> float:
        """Compare domains using sparse field difference."""
        csf = vls.CompareSparseField(domain1, domain2)
        if saveVisualization:
            mesh = vls.Mesh()
            csf.setOutputMesh(mesh)
        csf.apply()
        return csf.getSumSquaredDifferences()

    @staticmethod
    def _compareAreaAndSparseField(
        domain1: vls.Domain, domain2: vls.Domain, saveVisualization: bool = False
    ) -> float:
        """Compare domains using both area and sparse field metrics."""
        ca = vls.CompareArea(domain1, domain2)
        csf = vls.CompareSparseField(domain1, domain2)

        if saveVisualization:
            caMesh = vls.Mesh()
            csfMesh = vls.Mesh()
            ca.setOutputMesh(caMesh)
            csf.setOutputMesh(csfMesh)

        ca.apply()
        csf.apply()

        return ca.getAreaMismatch() + csf.getSumSquaredDifferences()
