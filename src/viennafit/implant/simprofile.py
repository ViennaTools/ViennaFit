"""
Utilities for extracting 1-D depth profiles from ViennaPS cell sets and for
building standard masked-substrate domains used in ion implantation simulations.

These functions bridge the ViennaPS simulation world with the 1-D SIMS fitting
tools in viennafit.implant.  viennaps is imported lazily so the rest of the
implant package works without it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._common import sims_depth_from_center


# ─── Lazy ViennaPS/ViennaLS compatibility helpers ─────────────────────────────

def _import_viennaps_2d():
    try:
        import viennaps.d2 as vps
        import viennaps as core
    except ImportError:
        raise ImportError(
            "ViennaPS domain builders require the viennaps Python bindings. "
            "Build ViennaPS with -DVIENNAPS_BUILD_PYTHON=ON and install the package."
        )
    return vps, core, core.ls


def _boolean_operation_enum():
    try:
        import viennals
    except ImportError:
        raise ImportError(
            "Masked-domain construction requires the viennals Python bindings, "
            "which are a ViennaPS dependency."
        )
    return viennals.BooleanOperationEnum


# ─── Profile extraction ───────────────────────────────────────────────────────

def extract_depth_profile(
    domain,
    label: str,
    depth_axis: int = 1,
    agg: str = "max",
    surface_position: float = 0.0,
    substrate_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1-D depth profile from a ViennaPS domain's cell set.

    Bins every cell by positive SIMS depth into the substrate, then aggregates
    the scalar field ``label`` across all lateral cells at each depth. ViennaPS
    examples place the Si surface at ``y = 0`` and the substrate at ``y < 0``;
    the returned depth is therefore ``surface_position - y`` by default.

    Parameters
    ----------
    domain     : viennaps.d2.Domain or d3.Domain object
    label      : name of the scalar field in the cell set (e.g. ``"P_total"``)
    depth_axis : cell-centre axis index for depth
                 (1 = y for 2-D simulations, 2 = z for 3-D, default 1)
    agg        : ``"max"`` — peak concentration across x at each depth (matches
                  a 1-D full-dose simulation at the beam centre)
                 ``"sum"`` — lateral sum (proportional to total dose per slice)
    surface_position : coordinate of the wafer surface along ``depth_axis`` [nm]
    substrate_only   : when True, ignore cells above the surface

    Returns
    -------
    depths : 1-D float array, sorted ascending (nm)
    values : 1-D float array, aggregated field values at each depth
             Returns two empty arrays if the field is absent.
    """
    cs    = domain.getCellSet()
    n     = cs.getNumberOfCells()
    delta = cs.getGridDelta()

    raw = cs.getScalarData(label)
    if raw is None or n == 0:
        return np.array([]), np.array([])

    values_all = np.asarray(raw, dtype=np.float64)

    depths = []
    values = []
    for idx in range(n):
        center = cs.getCellCenter(idx)
        depth = sims_depth_from_center(center, depth_axis, surface_position)
        if substrate_only and depth < -1.0e-12:
            continue
        depths.append(depth)
        values.append(values_all[idx])

    if not depths:
        return np.array([]), np.array([])

    # Round to nearest grid point so floating-point values collapse to keys
    depths_all = np.asarray(depths, dtype=np.float64)
    values_all = np.asarray(values, dtype=np.float64)
    depths_rounded = np.round(depths_all / delta) * delta

    # Aggregate per unique depth
    unique_depths = np.unique(depths_rounded)
    result_values = np.empty(len(unique_depths), dtype=np.float64)
    for i, d in enumerate(unique_depths):
        mask = depths_rounded == d
        if agg == "max":
            result_values[i] = values_all[mask].max()
        elif agg == "sum":
            result_values[i] = values_all[mask].sum()
        else:
            raise ValueError(f"agg must be 'max' or 'sum', got '{agg}'")

    return unique_depths, result_values


# ─── Domain builder ───────────────────────────────────────────────────────────

def build_blanket_substrate_domain(
    grid_delta: float,
    x_extent: float,
    top_space: float,
    substrate_depth: float,
    oxide_thickness: float = 0.,
) -> object:
    """
    Build a 2-D blanket (unmasked) substrate domain for calibrating against
    SIMS data from a bare-wafer implant.

    Geometry (2 or 3 level sets):
      * Si bulk   :  y ∈ [−substrate_depth, 0]
      * Screen SiO₂ (optional): y ∈ [0, oxide_thickness]
      * Air cover :  y ∈ [oxide_thickness, oxide_thickness + top_space]

    No mask layer is created, so the full surface receives dose.  This matches
    the conditions of a typical SIMS calibration measurement.

    Parameters
    ----------
    grid_delta      : cell size [nm]
    x_extent        : domain width [nm]  (reflective at x = ± x_extent/2)
    top_space       : air region above the surface [nm]
    substrate_depth : Si bulk depth [nm]
    oxide_thickness : screen-oxide thickness [nm]  (0 = bare Si surface)

    Returns
    -------
    viennaps.d2.Domain with cell set ready for implantation.

    Raises
    ------
    ImportError  if viennaps is not installed.
    """
    vps, _core, vls = _import_viennaps_2d()

    surface_top = max(oxide_thickness, 0.)
    bounds = [-0.5 * x_extent, 0.5 * x_extent,
              -substrate_depth,
              top_space + surface_top]
    bc = [_core.BoundaryType.REFLECTIVE_BOUNDARY,
          _core.BoundaryType.INFINITE_BOUNDARY]

    domain = vps.Domain(bounds, bc, grid_delta)

    def _ls():
        return vls.Domain(bounds, bc, grid_delta)

    def _make_plane(ls, origin, normal):
        vls.MakeGeometry(ls, vls.Plane(origin, normal)).apply()

    # Si substrate (bottom boundary + top surface at y = 0)
    ls = _ls()
    _make_plane(ls, [0., -substrate_depth], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.Si)

    ls = _ls()
    _make_plane(ls, [0., 0.], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.Si)

    # Screen oxide (optional — omit if oxide_thickness = 0)
    if oxide_thickness > 0.:
        ls = _ls()
        _make_plane(ls, [0., oxide_thickness], [0., 1.])
        domain.insertNextLevelSetAsMaterial(ls, _core.Material.SiO2)

    domain.generateCellSet(top_space, _core.Material.Air, True)
    domain.getCellSet().buildNeighborhood()
    return domain


def build_masked_substrate_domain(
    grid_delta: float,
    x_extent: float,
    top_space: float,
    substrate_depth: float,
    opening_width: float,
    mask_height: float,
    oxide_thickness: float,
) -> object:
    """
    Build a 2-D masked-substrate ViennaPS domain (Si / screen-oxide / hard-mask).

    Creates the same four-level-set geometry as ``pImplantManual.cpp``:

    * Level set 0: Si bottom boundary at y = −substrate_depth
    * Level set 1: Si/SiO₂ interface at y = 0  (Si bulk surface)
    * Level set 2: SiO₂/mask interface at y = oxide_thickness  (screen oxide top)
    * Level set 3: mask top with opening of width ``opening_width`` centred at x = 0

    A cell set is generated above the mask surface and ``buildNeighborhood()``
    is called.  The returned domain is ready for a ``Process`` call.

    Parameters
    ----------
    grid_delta       : cell size [nm]
    x_extent         : full domain width [nm]  (reflective at ± x_extent/2)
    top_space        : air region height above mask [nm]
    substrate_depth  : Si bulk depth below surface [nm]
    opening_width    : mask window width [nm]
    mask_height      : hard-mask thickness [nm]
    oxide_thickness  : screen-oxide thickness [nm]

    Returns
    -------
    viennaps.d2.Domain object with cell set ready for implantation.

    Raises
    ------
    ImportError  if viennaps is not installed.
    """
    vps, _core, vls = _import_viennaps_2d()

    bounds = [-0.5 * x_extent, 0.5 * x_extent,
              -substrate_depth,
              top_space + oxide_thickness + mask_height]
    bc = [_core.BoundaryType.REFLECTIVE_BOUNDARY,
          _core.BoundaryType.INFINITE_BOUNDARY]

    domain = vps.Domain(bounds, bc, grid_delta)

    def _ls():
        return vls.Domain(bounds, bc, grid_delta)

    def _make_plane(ls, origin, normal):
        vls.MakeGeometry(ls, vls.Plane(origin, normal)).apply()

    # Si substrate bottom
    ls = _ls()
    _make_plane(ls, [0., -substrate_depth], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.Si)

    # Si substrate top (surface at y = 0)
    ls = _ls()
    _make_plane(ls, [0., 0.], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.Si)

    # Screen oxide (y = 0 → y = oxide_thickness)
    ls = _ls()
    _make_plane(ls, [0., oxide_thickness], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.SiO2)

    # Hard mask with opening
    ls = _ls()
    _make_plane(ls, [0., oxide_thickness + mask_height], [0., 1.])
    domain.insertNextLevelSetAsMaterial(ls, _core.Material.Mask)

    window = _ls()
    vls.MakeGeometry(
        window,
        vls.Box(
            [-0.5 * opening_width, oxide_thickness - grid_delta],
            [ 0.5 * opening_width, oxide_thickness + mask_height + grid_delta],
        ),
    ).apply()
    bool_op = _boolean_operation_enum()
    domain.applyBooleanOperation(
        window, bool_op.RELATIVE_COMPLEMENT)

    domain.generateCellSet(top_space, _core.Material.Air, True)
    domain.getCellSet().buildNeighborhood()
    return domain
