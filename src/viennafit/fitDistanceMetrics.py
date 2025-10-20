from typing import Callable
import viennals2d as vls
import os


class DistanceMetric:
    """Factory class for distance metrics used in optimization."""

    AVAILABLE_METRICS = ["CA", "CSF", "CSF-IS", "CNB", "CA+CSF", "CA+CNB", "CCD", "CCH"]

    @staticmethod
    def create(
        metricName: str,
        multiDomain: bool = False,
        criticalDimensionRanges: list[dict] = None,
        sparseFieldExpansionWidth: int = 200,
    ) -> Callable:
        """
        Create a distance metric function based on metric name.

        Args:
            metricName: Name of the distance metric to use
            multiDomain: If True, returns function for comparing multiple domain pairs
            criticalDimensionRanges: List of range configurations for CCD metric. Each dict should have:
                - 'axis': 'x' or 'y' (the axis to scan along)
                - 'min': minimum value of the range
                - 'max': maximum value of the range
                - 'findMaximum': True to find maximum, False to find minimum
                Example: [{'axis': 'x', 'min': -5, 'max': 5, 'findMaximum': True}]
            sparseFieldExpansionWidth: Expansion width for CSF metric (default: 200).
                Applied to target level set before comparison to ensure sufficient overlap.

        Returns:
            For single domain: Callable[[vls.Domain, vls.Domain, bool, str], float]
            For multi domain: Callable[[dict[str, vls.Domain], dict[str, vls.Domain], bool, str], float]
        """
        if multiDomain:
            if metricName == "CA+CSF":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path,
                    lambda d1, d2, sv, wp: DistanceMetric._compareAreaAndSparseField(
                        d1, d2, sv, wp, sparseFieldExpansionWidth
                    )
                )
            elif metricName == "CA":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path, DistanceMetric._compareArea
                )
            elif metricName == "CSF":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path,
                    lambda d1, d2, sv, wp: DistanceMetric._compareSparseField(
                        d1, d2, sv, wp, sparseFieldExpansionWidth
                    )
                )
            elif metricName == "CSF-IS":
                # Create caching wrapper for CSF-IS to expand target only once per domain
                expandedTargetCache = {}

                def cached_csf_is_single(d1, d2, sv, wp):
                    """Cached CSF-IS for single domain comparison."""
                    import time as time_module
                    cacheKey = id(d2)

                    if cacheKey not in expandedTargetCache:
                        # First call: expand target and cache it (exclude from timing)
                        expandStartTime = time_module.time()
                        expandedTarget = vls.Domain(d2)
                        vls.Expand(expandedTarget, sparseFieldExpansionWidth).apply()
                        expandTime = time_module.time() - expandStartTime
                        expandedTargetCache[cacheKey] = expandedTarget
                        # print(f"CSF-IS: Expanded target domain (took {expandTime:.3f}s, excluded from metric time)")

                    # Use cached expanded target
                    return DistanceMetric._compareSparseFieldIterateSample(
                        d1, expandedTargetCache[cacheKey], sv, wp, sparseFieldExpansionWidth
                    )

                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path, cached_csf_is_single
                )
            elif metricName == "CNB":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path, DistanceMetric._compareNarrowBand
                )
            elif metricName == "CA+CNB":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path, DistanceMetric._compareAreaAndNarrowBand
                )
            elif metricName == "CCD":
                if criticalDimensionRanges is None:
                    raise ValueError("criticalDimensionRanges must be provided for CCD metric")
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path,
                    lambda d1, d2, sv, wp: DistanceMetric._compareCriticalDimensions(
                        d1, d2, sv, wp, criticalDimensionRanges
                    )
                )
            elif metricName == "CCH":
                return lambda domains1, domains2, saveVis, path: DistanceMetric._compareMultipleDomains(
                    domains1, domains2, saveVis, path, DistanceMetric._compareChamfer
                )
            else:
                raise ValueError(
                    f"Invalid distance metric: {metricName}. "
                    f"Available options are: {', '.join(DistanceMetric.AVAILABLE_METRICS)}"
                )
        else:
            if metricName == "CA+CSF":
                return lambda d1, d2, sv, wp: DistanceMetric._compareAreaAndSparseField(
                    d1, d2, sv, wp, sparseFieldExpansionWidth
                )
            elif metricName == "CA":
                return DistanceMetric._compareArea
            elif metricName == "CSF":
                return lambda d1, d2, sv, wp: DistanceMetric._compareSparseField(
                    d1, d2, sv, wp, sparseFieldExpansionWidth
                )
            elif metricName == "CSF-IS":
                # Create caching wrapper for CSF-IS to expand target only once
                expandedTargetCache = {}

                def cached_csf_is(d1, d2, sv, wp):
                    """Cached CSF-IS that expands target (d2) only once."""
                    import time as time_module
                    cacheKey = id(d2)

                    if cacheKey not in expandedTargetCache:
                        # First call: expand target and cache it (exclude from timing)
                        expandStartTime = time_module.time()
                        expandedTarget = vls.Domain(d2)
                        vls.Expand(expandedTarget, sparseFieldExpansionWidth).apply()
                        expandTime = time_module.time() - expandStartTime
                        expandedTargetCache[cacheKey] = expandedTarget
                        # print(f"CSF-IS: Expanded target domain (took {expandTime:.3f}s, excluded from metric time)")

                    # Use cached expanded target
                    # Note: d1 is sample (will be iterated), cached target is expanded
                    return DistanceMetric._compareSparseFieldIterateSample(
                        d1, expandedTargetCache[cacheKey], sv, wp, sparseFieldExpansionWidth
                    )

                return cached_csf_is
            elif metricName == "CNB":
                return DistanceMetric._compareNarrowBand
            elif metricName == "CA+CNB":
                return DistanceMetric._compareAreaAndNarrowBand
            elif metricName == "CCD":
                if criticalDimensionRanges is None:
                    raise ValueError("criticalDimensionRanges must be provided for CCD metric")
                return lambda d1, d2, sv, wp: DistanceMetric._compareCriticalDimensions(
                    d1, d2, sv, wp, criticalDimensionRanges
                )
            elif metricName == "CCH":
                return DistanceMetric._compareChamfer
            else:
                raise ValueError(
                    f"Invalid distance metric: {metricName}. "
                    f"Available options are: {', '.join(DistanceMetric.AVAILABLE_METRICS)}"
                )

    @staticmethod
    def _compareArea(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
    ) -> float:
        """Compare domains using area mismatch."""
        ca = vls.CompareArea(domain1, domain2)
        if saveVisualization:
            mesh = vls.Mesh()
            ca.setOutputMesh(mesh)
        ca.apply()

        if saveVisualization:
            # Save mesh to progress directory with evaluation counter
            caPath = os.path.join(
                f"{writePath}-CA.vtu",
            )
            vls.VTKWriter(mesh, caPath).apply()

        return ca.getAreaMismatch()

    @staticmethod
    def _compareSparseField(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
        expansionWidth: int = 200,
    ) -> float:
        """Compare domains using sparse field difference."""
        csf = vls.CompareSparseField(domain1, domain2)
        csf.setExpandedLevelSetWidth(expansionWidth)
        if saveVisualization:
            mesh = vls.Mesh()
            csf.setOutputMesh(mesh)
        csf.apply()

        if saveVisualization:
            # Save mesh to progress directory with evaluation counter
            csfPath = os.path.join(
                f"{writePath}-CSF.vtp",
            )
            vls.VTKWriter(mesh, csfPath).apply()

        return csf.getSumSquaredDifferences()

    @staticmethod
    def _compareSparseFieldIterateSample(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
        expansionWidth: int = 200,
    ) -> float:
        """
        Compare domains using sparse field difference with SWAPPED roles.

        In this variant:
        - domain2 (target) is EXPANDED and used as reference
        - domain1 (sample/result) is ITERATED (reduced to sparse field)

        This is the INVERSE of the standard CSF metric, useful when you want
        to iterate over the result/sample points rather than the target points.

        Args:
            domain1: Sample domain (will be iterated - reduced to sparse field)
            domain2: Target domain (will be expanded as reference)
            saveVisualization: Whether to save visualization files
            writePath: Path for saving visualization files
            expansionWidth: Expansion width for target domain (default: 200)

        Returns:
            Sum of squared differences
        """
        # SWAP: Pass domain2 as expanded, domain1 as iterated
        csf = vls.CompareSparseField(domain2, domain1)
        csf.setExpandedLevelSetWidth(expansionWidth)
        if saveVisualization:
            mesh = vls.Mesh()
            csf.setOutputMesh(mesh)
        csf.apply()

        if saveVisualization:
            # Save mesh to progress directory with evaluation counter
            csfPath = os.path.join(
                f"{writePath}-CSF-IS.vtp",
            )
            vls.VTKWriter(mesh, csfPath).apply()

        return csf.getSumSquaredDifferences()

    @staticmethod
    def _compareAreaAndSparseField(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
        expansionWidth: int = 200,
    ) -> float:
        """Compare domains using both area and sparse field metrics."""
        ca = vls.CompareArea(domain1, domain2)
        csf = vls.CompareSparseField(domain1, domain2)
        csf.setExpandedLevelSetWidth(expansionWidth)

        if saveVisualization:
            caMesh = vls.Mesh()
            csfMesh = vls.Mesh()
            ca.setOutputMesh(caMesh)
            csf.setOutputMesh(csfMesh)

        ca.apply()
        csf.apply()

        if saveVisualization:
            # Save meshes to progress directory with evaluation counter
            caPath = os.path.join(
                f"{writePath}-CA.vtu",
            )
            csfPath = os.path.join(
                f"{writePath}-CSF.vtp",
            )
            vls.VTKWriter(caMesh, caPath).apply()
            vls.VTKWriter(csfMesh, csfPath).apply()

        return ca.getAreaMismatch() + csf.getSumSquaredDifferences()

    @staticmethod
    def _compareNarrowBand(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
    ) -> float:
        """Compare domains using narrow band difference."""
        cnb = vls.CompareNarrowBand(domain1, domain2)
        if saveVisualization:
            mesh = vls.Mesh()
            cnb.setOutputMesh(mesh)
        cnb.apply()

        if saveVisualization:
            # Save mesh to progress directory with evaluation counter
            cnbPath = f"{writePath}-CNB.vtp"
            vls.VTKWriter(mesh, cnbPath).apply()

        return cnb.getSumSquaredDifferences()

    @staticmethod
    def _compareAreaAndNarrowBand(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
    ) -> float:
        """Compare domains using both area and narrow band metrics."""
        ca = vls.CompareArea(domain1, domain2)
        cnb = vls.CompareNarrowBand(domain1, domain2)

        if saveVisualization:
            caMesh = vls.Mesh()
            cnbMesh = vls.Mesh()
            ca.setOutputMesh(caMesh)
            cnb.setOutputMesh(cnbMesh)

        ca.apply()
        cnb.apply()

        if saveVisualization:
            # Save meshes to progress directory with evaluation counter
            caPath = f"{writePath}-CA.vtu"
            cnbPath = f"{writePath}-CNB.vtp"
            vls.VTKWriter(caMesh, caPath).apply()
            vls.VTKWriter(cnbMesh, cnbPath).apply()

        return ca.getAreaMismatch() + cnb.getSumSquaredDifferences()

    @staticmethod
    def _compareCriticalDimensions(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
        ranges: list[dict] = None,
    ) -> float:
        """
        Compare domains using critical dimensions at user-specified measurement points.

        Args:
            domain1: First domain to compare
            domain2: Second domain to compare
            saveVisualization: Whether to save visualization files
            writePath: Path for saving visualization files
            ranges: List of range configurations. Each dict should have:
                - 'axis': 'x' or 'y' (the axis to scan along)
                - 'min': minimum value of the range
                - 'max': maximum value of the range
                - 'findMaximum': True to find maximum, False to find minimum

        Returns:
            RMSE of the critical dimension differences
        """
        if ranges is None or len(ranges) == 0:
            raise ValueError("At least one range must be specified for critical dimensions comparison")

        ccd = vls.CompareCriticalDimensions(domain1, domain2)

        # Add user-specified measurement ranges
        for rangeConfig in ranges:
            axis = rangeConfig.get('axis', '').lower()
            minVal = rangeConfig.get('min')
            maxVal = rangeConfig.get('max')
            findMaximum = rangeConfig.get('findMaximum', True)

            if axis not in ['x', 'y']:
                raise ValueError(f"Invalid axis '{axis}'. Must be 'x' or 'y'")
            if minVal is None or maxVal is None:
                raise ValueError("Range must specify 'min' and 'max' values")

            if axis == 'x':
                ccd.addXRange(float(minVal), float(maxVal), bool(findMaximum))
            else:  # axis == 'y'
                ccd.addYRange(float(minVal), float(maxVal), bool(findMaximum))

        if saveVisualization:
            mesh = vls.Mesh()
            ccd.setOutputMesh(mesh)

        ccd.apply()

        if saveVisualization:
            # Save mesh to progress directory with evaluation counter
            ccdPath = f"{writePath}-CCD.vtp"
            vls.VTKWriter(mesh, ccdPath).apply()

        return ccd.getRMSE()

    @staticmethod
    def _compareChamfer(
        domain1: vls.Domain,
        domain2: vls.Domain,
        saveVisualization: bool = False,
        writePath: str = None,
    ) -> float:
        """
        Compare domains using Chamfer distance between surfaces.

        The Chamfer distance is a bidirectional metric that measures the average
        nearest-neighbor distance between two point sets (surfaces). It returns
        the RMS Chamfer distance which provides a robust measure of shape similarity.

        Args:
            domain1: Target domain (reference)
            domain2: Sample domain (to compare)
            saveVisualization: Whether to save visualization files
            writePath: Path for saving visualization files

        Returns:
            RMS Chamfer distance between the two surfaces
        """
        cch = vls.CompareChamfer(domain1, domain2)

        if saveVisualization:
            targetMesh = vls.Mesh()
            sampleMesh = vls.Mesh()
            cch.setOutputMeshTarget(targetMesh)
            cch.setOutputMeshSample(sampleMesh)

        cch.apply()

        if saveVisualization:
            # Save meshes to show distance distributions on both surfaces
            targetPath = f"{writePath}-CCH-target.vtp"
            samplePath = f"{writePath}-CCH-sample.vtp"
            vls.VTKWriter(targetMesh, targetPath).apply()
            vls.VTKWriter(sampleMesh, samplePath).apply()

        return cch.getRMSChamferDistance()

    @staticmethod
    def _compareMultipleDomains(
        resultDomains: dict[str, vls.Domain],
        targetDomains: dict[str, vls.Domain],
        saveVisualization: bool = False,
        writePath: str = None,
        singleDomainMetric: Callable = None,
    ) -> float:
        """
        Compare multiple domain pairs using specified single domain metric.

        Args:
            resultDomains: Dictionary of result domains keyed by domain name
            targetDomains: Dictionary of target domains keyed by domain name
            saveVisualization: Whether to save visualization files
            writePath: Base path for saving visualization files
            singleDomainMetric: Single domain comparison function to use

        Returns:
            Sum of all individual domain pair comparisons
        """
        if singleDomainMetric is None:
            raise ValueError("Single domain metric function must be provided")

        totalDistance = 0.0

        # Ensure all result domains have corresponding target domains
        for domainName in resultDomains.keys():
            if domainName not in targetDomains:
                raise ValueError(f"No target domain found for result domain '{domainName}'")

        # Compare each domain pair
        for domainName in resultDomains.keys():
            resultDomain = resultDomains[domainName]
            targetDomain = targetDomains[domainName]

            # Create domain-specific write path if visualization is enabled
            domainWritePath = None
            if saveVisualization and writePath:
                domainWritePath = f"{writePath}-{domainName}"

            # Calculate distance for this domain pair
            distance = singleDomainMetric(
                resultDomain, targetDomain, saveVisualization, domainWritePath
            )

            totalDistance += distance

        return totalDistance
