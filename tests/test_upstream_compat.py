"""
Upstream compatibility checks for ViennaFit against ViennaPS / ViennaLS.

ViennaFit depends on a small, specific slice of the ViennaPS/ViennaLS Python
API. These checks guard exactly that slice so an upstream rename, removal or
signature change is caught here instead of in a user's optimization run.

Two tiers:
  1. Contract  - every ViennaPS/ViennaLS symbol and method ViennaFit relies on
                 must exist. Fast, deterministic, no simulation required.
  2. Functional- build real 2D level sets and run every distance metric through
                 ViennaFit's own ``DistanceMetric.create(...)`` entry point.
                 Catches behaviour/signature breaks the contract tier can't see.

Note: ViennaFit reaches ViennaLS through ``viennaps.ls`` (ViennaPS owns the
ViennaLS version). These checks deliberately do the same, so they validate the
exact module object ViennaFit uses at runtime.

Run either way:
    python tests/test_upstream_compat.py     # plain, no pytest needed (CI)
    pytest tests/test_upstream_compat.py     # if pytest is available
"""

import viennaps as vps
from viennaps import ls as vls

# --- Symbols ViennaFit imports/uses -----------------------------------------

VPS_SYMBOLS = ["Domain", "Reader", "Writer", "setDimension", "ls"]

VLS_SYMBOLS = [
    "Domain",
    "Expand",
    "Mesh",
    "ToMesh",
    "ToSurfaceMesh",
    "VTKWriter",
    "Writer",
    "Reader",
    "setDimension",
    "CompareArea",
    "CompareSparseField",
    "CompareNarrowBand",
    "CompareCriticalDimensions",
    "CompareChamfer",
]

# Methods ViennaFit calls on each ViennaLS comparison class.
VLS_COMPARE_METHODS = {
    "CompareArea": ["apply", "setOutputMesh", "getAreaMismatch"],
    "CompareSparseField": [
        "apply",
        "setOutputMesh",
        "setExpandedLevelSetWidth",
        "getRMSE",
        "getSumSquaredDifferences",
    ],
    "CompareNarrowBand": [
        "apply",
        "setOutputMesh",
        "getRMSE",
        "getSumSquaredDifferences",
    ],
    "CompareCriticalDimensions": [
        "apply",
        "setOutputMesh",
        "addXRange",
        "addYRange",
        "getRMSE",
    ],
    "CompareChamfer": [
        "apply",
        "getRMSChamferDistance",
        "setOutputMeshSample",
        "setOutputMeshTarget",
    ],
}


def test_vps_symbols_present():
    missing = [name for name in VPS_SYMBOLS if not hasattr(vps, name)]
    assert not missing, f"ViennaPS missing symbols ViennaFit needs: {missing}"


def test_vls_symbols_present():
    missing = [name for name in VLS_SYMBOLS if not hasattr(vls, name)]
    assert not missing, f"ViennaLS missing symbols ViennaFit needs: {missing}"


def test_vls_is_reexported_external_package():
    # ViennaFit's "ViennaPS owns ViennaLS" assumption: viennaps.ls must be the
    # one installed viennals package, not a vendored private copy.
    import viennals as standalone

    assert vls is standalone, "viennaps.ls is no longer the external viennals package"


def test_vls_compare_methods_present():
    problems = []
    for cls_name, methods in VLS_COMPARE_METHODS.items():
        cls = getattr(vls, cls_name, None)
        if cls is None:
            problems.append(f"{cls_name} (class missing)")
            continue
        for m in methods:
            if not hasattr(cls, m):
                problems.append(f"{cls_name}.{m}")
    assert not problems, f"ViennaLS comparison API changed: {problems}"


# --- Functional tier ---------------------------------------------------------


def _make_pair():
    """Two slightly different 2D level sets, as ViennaFit would compare them."""
    vls.setDimension(2)
    bounds = [-10.0, 10.0, -10.0, 10.0]
    bcs = [
        vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ]
    grid_delta = 0.5
    d1 = vls.Domain(bounds, bcs, grid_delta)
    d2 = vls.Domain(bounds, bcs, grid_delta)
    vls.MakeGeometry(d1, vls.Sphere([0.0, 0.0], 5.0)).apply()
    vls.MakeGeometry(d2, vls.Sphere([0.0, 0.0], 5.5)).apply()
    return d1, d2


def test_distance_metrics_run_end_to_end():
    from viennafit.fitDistanceMetrics import DistanceMetric

    d1, d2 = _make_pair()
    # Single-domain metrics that need no extra configuration.
    for name in ["CA", "CSF", "CSF-IS", "CNB", "CA+CSF", "CA+CNB", "CCH"]:
        metric = DistanceMetric.create(name)
        value = metric(d1, d2, False, "")
        assert isinstance(value, (int, float)), f"{name} returned {type(value)}"
        assert value == value, f"{name} returned NaN"  # NaN != NaN

    # CCD requires range configuration.
    ccd = DistanceMetric.create(
        "CCD",
        criticalDimensionRanges=[
            {"axis": "x", "min": -5.0, "max": 5.0, "findMaximum": True}
        ],
    )
    value = ccd(d1, d2, False, "")
    assert isinstance(value, (int, float)), f"CCD returned {type(value)}"


def test_levelset_round_trip(tmp_path=None):
    import tempfile
    import os

    d1, _ = _make_pair()
    out_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    path = os.path.join(out_dir, "roundtrip.lvst")
    vls.Writer(d1, path).apply()
    assert os.path.exists(path), "ViennaLS Writer produced no file"
    reloaded = vls.Domain(2)
    vls.Reader(reloaded, path).apply()
    assert reloaded.getNumberOfPoints() > 0, "ViennaLS Reader produced empty domain"


# --- Plain runner (no pytest dependency) ------------------------------------

if __name__ == "__main__":
    import traceback

    print(f"ViennaPS {vps.__version__}  |  ViennaLS {vls.__version__}")
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
    if failures:
        raise SystemExit(f"\n{failures}/{len(tests)} compatibility checks FAILED")
    print(f"\nAll {len(tests)} compatibility checks passed.")
