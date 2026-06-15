"""
Anneal model calibration against post-anneal SIMS and sheet resistance.

``AnnealCalibrator`` runs the ViennaPS ``Anneal`` process inside each optimizer
call, extracting the post-anneal depth profile and computing the sheet
resistance.  It accepts the same optimizer backends as ``PearsonIVFitter`` and
``SimulationCalibrator`` (scipy / dlib / nevergrad / cma).

Three calibration targets (all optional individually):
  1. Post-anneal SIMS depth profile — constrains diffusivity (D0, Ea)
  2. Measured sheet resistance (Rsh, Ω/□) — constrains solid solubility (C0, Ea_ss)

Fitted parameters
-----------------
  Always:   log10(annealD0 [nm²/s]),  annealEa [eV]
  If rsh:   log10(solidSolC0 [nm⁻³]), solidSolEa [eV]

Concentration units
-------------------
ViennaPS cell-set fields are stored in **nm⁻³** (domain length unit = nm).
SIMS profiles normalised to peak=1 are rescaled using ``dose_cm2`` before being
written; the conversion is 1 cm⁻³ = 1e-21 nm⁻³.

Sheet resistance
----------------
Rsh is computed by ``viennaps.d2.SheetResistance`` (or ``viennacs.d2``), which
lives in ViennaCS/ViennaPS where it belongs — ViennaFit does not duplicate the
Masetti mobility model.

Typical usage
-------------
  from viennafit.implant import (SimsProfile, AnnealCalibrator,
                                  build_blanket_substrate_domain)
  import functools

  pre   = SimsProfile.from_csv("sims_5keV_P_Si.csv",          label="pre-anneal")
  post  = SimsProfile.from_csv("sims_5keV_P_Si_postanneal.csv", label="post-anneal")

  domain_factory = functools.partial(
      build_blanket_substrate_domain,
      grid_delta=1.0, x_extent=50., top_space=10.,
      substrate_depth=80., oxide_thickness=2.,
  )
  cal = AnnealCalibrator(
      pre_anneal_sims  = pre,
      post_anneal_sims = post,
      rsh_target       = 1200.,      # Ω/□  —  None to skip Rsh term
      anneal_temp_C    = 1000.,
      anneal_time_s    = 30.,
      domain_factory   = domain_factory,
      dose_cm2         = 1e14,
  )
  result = cal.fit(optimizer="scipy", n_budget=200)
  result.print_summary()
  print(result.to_config_string())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._common import (
    OPTIMIZERS,
    residual_loss,
    r2_log,
    rmse_log,
    run_optimizer,
    sims_depth_from_center,
)
from .simsfit import SimsProfile
from .simprofile import extract_depth_profile

# ── Cell-set initialisation from a 1-D SIMS profile ──────────────────────────

def _write_cell_scalar_data(cell_set, label: str, values: List[float]) -> None:
    """
    Write scalar data through the best API exposed by the installed bindings.

    ViennaCS exposes ``DenseCellSet.setScalarData`` directly. Some ViennaPS
    builds only expose ``getCellGrid().getCellData().insertNextScalarData``;
    in those builds ``getScalarData`` returns a copy, so in-place list edits do
    not update the cell set.
    """
    if hasattr(cell_set, "setScalarData"):
        cell_set.addScalarData(label, 0.0)
        cell_set.setScalarData(label, values)
        return

    labels = set(cell_set.getScalarDataLabels())
    if label in labels:
        raise RuntimeError(
            f"Cannot overwrite existing cell-set scalar '{label}' because this "
            "ViennaPS Python binding does not expose DenseCellSet.setScalarData."
        )

    cell_data = cell_set.getCellGrid().getCellData()
    cell_data.insertNextScalarData(values, label)


def init_domain_from_sims(
    domain,
    species: str,
    depth_nm: np.ndarray,
    conc: np.ndarray,
    dose_cm2: float = 1e14,
    depth_axis: int = 1,
    surface_position: float = 0.0,
    substrate_only: bool = True,
) -> None:
    """
    Write a 1-D concentration profile into a ViennaPS cell set.

    The concentration field ``{species}_total`` is created (or overwritten) in
    the cell set by interpolating the supplied profile at each cell's depth
    coordinate. ViennaPS uses ``y < 0`` for the Si substrate in the standard
    examples; SIMS depth is positive into the substrate, so the default mapping
    is ``depth_nm = 0 - y``.

    If ``conc`` appears to be normalised (peak ≤ 2), it is rescaled to absolute
    units [nm⁻³] using ``dose_cm2``:

        dose [cm⁻²] = ∫ C_abs(z) dz [nm] × 1e-7 [cm/nm]
        → C_abs [nm⁻³] = dose / (∫ conc(z) dz [nm] × 1e-7) × 1e-21

    The resulting field is in **nm⁻³**, consistent with ViennaPS internal units.

    Parameters
    ----------
    domain   : viennaps.d2.Domain with a generated cell set
    species  : dopant label, e.g. ``"P"``
    depth_nm : 1-D depth array [nm], sorted ascending, starting from 0
    conc     : concentration values at each depth; either absolute [cm⁻³]
               or normalised (peak ≈ 1)
    dose_cm2 : implant dose [cm⁻²], used to rescale a normalised profile
    depth_axis : coordinate axis used as depth (1 = y in 2-D)
    surface_position : wafer-surface coordinate along ``depth_axis`` [nm]
    substrate_only : when True, write zero to cells above the surface
    """
    depth_nm = np.asarray(depth_nm, dtype=np.float64)
    conc     = np.asarray(conc,     dtype=np.float64)

    peak = conc.max()
    if 0.0 < peak <= 2.0:
        # Normalised profile → absolute cm⁻³
        integral_nm = float(np.trapz(conc, depth_nm))
        if integral_nm > 0.0:
            conc = conc * dose_cm2 / (integral_nm * 1e-7)

    # cm⁻³ → nm⁻³  (ViennaPS internal unit)
    conc_nm3 = conc * 1e-21

    cs    = domain.getCellSet()
    n     = cs.getNumberOfCells()
    label = f"{species}_total"

    concentrations: List[float] = []
    for i in range(n):
        center = cs.getCellCenter(i)
        depth = sims_depth_from_center(center, depth_axis, surface_position)
        c = (float(np.interp(depth, depth_nm, conc_nm3, left=0., right=0.))
             if (not substrate_only or depth >= 0.0) else 0.0)
        concentrations.append(c)

    _write_cell_scalar_data(cs, label, concentrations)


# ── Anneal fit result ─────────────────────────────────────────────────────────

@dataclass
class AnnealFitResult:
    """
    Best-fit result from ``AnnealCalibrator.fit()``.

    All length units: nm.  Diffusivity D0: nm²/s (ViennaPS internal).
    Concentrations in nm⁻³ (ViennaPS internal); config strings show the
    conversion to cm⁻³ where useful.
    """
    annealD0:       float
    annealEa:       float
    solidSolC0:     Optional[float] = None   # nm⁻³
    solidSolEa:     Optional[float] = None   # eV

    # Stored simulated post-anneal profile (nm depths, normalised values)
    sim_depths:     Optional[np.ndarray] = field(default=None, repr=False)
    sim_profile:    Optional[np.ndarray] = field(default=None, repr=False)

    # Fit quality
    r2_log:         Optional[float] = None
    rmse_log:       Optional[float] = None
    rsh_computed:   Optional[float] = None   # Ω/□
    rsh_target:     Optional[float] = None   # Ω/□
    rsh_error_pct:  Optional[float] = None   # %

    n_eval:         int = 0
    optimizer:      str = ""

    # ── serialisation ────────────────────────────────────────────────────────

    def to_config_string(self) -> str:
        """Return a config.txt block for the fitted anneal parameters."""
        lines = [
            "# ── Anneal parameters (calibrated) ─────────────────────",
            f"annealParameterSource=manual",
            f"annealD0={self.annealD0:.6e}",
            f"annealEa={self.annealEa:.4f}",
        ]
        if self.solidSolC0 is not None:
            lines += [
                "annealSolidActivation=1",
                f"annealSolidSolubilityC0={self.solidSolC0:.6e}   # nm⁻³  "
                f"(= {self.solidSolC0 * 1e21:.3e} cm⁻³)",
                f"annealSolidSolubilityEa={self.solidSolEa:.4f}",
            ]
        return "\n".join(lines)

    # ── display ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print(f"  annealD0       = {self.annealD0:.3e} nm²/s  "
              f"(= {self.annealD0 * 1e-14:.3e} cm²/s)")
        print(f"  annealEa       = {self.annealEa:.4f} eV")
        if self.solidSolC0 is not None:
            print(f"  solidSolC0     = {self.solidSolC0:.3e} nm⁻³  "
                  f"(= {self.solidSolC0 * 1e21:.3e} cm⁻³)")
            print(f"  solidSolEa     = {self.solidSolEa:.4f} eV")
        if self.r2_log is not None:
            print(f"  R² (log)       = {self.r2_log:.4f}")
            print(f"  RMSE (log)     = {self.rmse_log:.4f}")
        if self.rsh_computed is not None:
            sign = "↑" if self.rsh_computed > self.rsh_target else "↓"
            print(f"  Rsh_sim        = {self.rsh_computed:.1f} Ω/□  "
                  f"(target {self.rsh_target:.1f},  "
                  f"err {self.rsh_error_pct:+.1f}% {sign})")
        print(f"  n_eval         = {self.n_eval}  [{self.optimizer}]")

    def plot(self, pre_anneal: SimsProfile, post_anneal: SimsProfile,
             ax=None, show: bool = True):
        """
        Plot pre-anneal SIMS, post-anneal SIMS target, and simulated profile.

        Parameters
        ----------
        pre_anneal  : SimsProfile before annealing
        post_anneal : SimsProfile after annealing (calibration target)
        ax          : matplotlib Axes (optional; created if None)
        show        : call plt.show() when done
        """
        import matplotlib.pyplot as plt
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.semilogy(pre_anneal.depth, pre_anneal.intensity,
                    "o--", color="steelblue", ms=4, lw=1,
                    label=pre_anneal.label or "Pre-anneal SIMS")
        ax.semilogy(post_anneal.depth, post_anneal.intensity,
                    "s-", color="firebrick", ms=5, lw=1.5,
                    label=post_anneal.label or "Post-anneal SIMS")
        if self.sim_depths is not None and self.sim_profile is not None:
            ax.semilogy(self.sim_depths, self.sim_profile,
                        "-", color="darkorange", lw=2,
                        label=f"Simulation [{self.optimizer}]")

        ax.set_xlabel("Depth (nm)")
        ax.set_ylabel("Normalised intensity")
        ax.set_ylim(1e-4, 5)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)

        info = f"D₀={self.annealD0:.2e} nm²/s, Eₐ={self.annealEa:.2f} eV"
        if self.rsh_computed is not None:
            info += f"\nRsh={self.rsh_computed:.0f} Ω/□ (target {self.rsh_target:.0f})"
        ax.set_title(info, fontsize=9)

        if show and fig is not None:
            plt.tight_layout()
            plt.show()


# ── Default parameter bounds ──────────────────────────────────────────────────

# Diffusivity D0 [nm²/s] — reference for P in Si: 4e13 nm²/s (log10 ≈ 13.6)
# Solid solubility C0 [nm⁻³] — reference for P in Si: 42 nm⁻³ (log10 ≈ 1.6)
_DEFAULT_ANNEAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "log10_D0":    (8.0, 18.0),   # log10(nm²/s)
    "annealEa":    (0.5,  5.5),   # eV
    "log10_C0_ss": (-2.0, 5.0),   # log10(nm⁻³)
    "Ea_ss":       (0.0,  2.0),   # eV
}


# ── Anneal calibrator ─────────────────────────────────────────────────────────

class AnnealCalibrator:
    """
    Calibrate ViennaPS ``Anneal`` parameters against post-anneal SIMS and,
    optionally, a measured sheet resistance.

    Each optimizer call:
      1. Builds a fresh domain via ``domain_factory()``.
      2. Initialises the cell set with the pre-anneal SIMS profile (scaled to
         ``dose_cm2`` in nm⁻³).
      3. Runs ``Anneal`` with the trial (D0, Ea, [C0_ss, Ea_ss]) parameters.
      4. Extracts the post-anneal depth profile and computes Rsh.
      5. Returns a combined residual vs. the calibration targets.

    Parameters
    ----------
    pre_anneal_sims  : SimsProfile — post-implant, pre-anneal SIMS measurement
    post_anneal_sims : SimsProfile — post-anneal SIMS measurement (target)
    rsh_target       : measured sheet resistance [Ω/□]; ``None`` disables the
                       Rsh term and skips fitting solid-solubility parameters
    anneal_temp_C    : anneal temperature [°C]
    anneal_time_s    : anneal duration [s]
    domain_factory   : callable ``() → domain`` — builds a fresh blanket domain
                       each call (use ``build_blanket_substrate_domain``).
    species          : dopant element label (default ``"P"``)
    dose_cm2         : implant dose [cm⁻²] used to scale the SIMS profile
    rsh_weight       : fraction of the objective attributed to Rsh residual
                       (ignored when ``rsh_target`` is None; default 0.4)
    log_fit          : minimise log-space residuals for the profile (default True)
    min_log10        : floor for log conversion of normalised intensity
    depth_agg        : ``"max"`` — peak across x per depth slice (default)
    """

    def __init__(
        self,
        pre_anneal_sims:  SimsProfile,
        post_anneal_sims: SimsProfile,
        rsh_target:       Optional[float],
        anneal_temp_C:    float,
        anneal_time_s:    float,
        domain_factory:   Callable,
        *,
        species:    str   = "P",
        dose_cm2:   float = 1e14,
        rsh_weight: float = 0.4,
        log_fit:    bool  = True,
        min_log10:  float = -6.0,
        depth_agg:  str   = "max",
    ):
        self.pre_anneal   = pre_anneal_sims
        self.post_anneal  = post_anneal_sims.normalize()
        self.rsh_target   = rsh_target
        self.temp_K       = anneal_temp_C + 273.15
        self.time_s       = anneal_time_s
        self.domain_factory = domain_factory
        self.species      = species
        self.dose_cm2     = dose_cm2
        self.rsh_weight   = rsh_weight if rsh_target is not None else 0.0
        self.log_fit      = log_fit
        self.floor        = 10.0 ** min_log10
        self.depth_agg    = depth_agg
        self._fit_ss      = rsh_target is not None
        self._n_eval      = 0

        self._label_total  = f"{species}_total"
        self._label_active = f"{species}_active"

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        bounds:   Optional[Dict[str, Tuple[float, float]]] = None,
        n_budget: int = 200,
        optimizer: str = "scipy",
        seed:     int = 42,
    ) -> AnnealFitResult:
        """
        Run the calibration.

        Parameters
        ----------
        bounds    : override default parameter bounds (see ``_DEFAULT_ANNEAL_BOUNDS``)
        n_budget  : number of simulation calls (budget); keep ≤ 300 for speed
        optimizer : ``"scipy"`` | ``"dlib"`` | ``"nevergrad"`` | ``"cma"``
        seed      : random seed

        Returns
        -------
        ``AnnealFitResult`` — ``to_config_string()`` and ``plot()`` work
        without any further simulation calls.
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}")

        # Merge user bounds with defaults
        merged = {**_DEFAULT_ANNEAL_BOUNDS, **(bounds or {})}

        try:
            import viennaps.d2 as vps
            import viennaps as _core
        except ImportError:
            raise ImportError(
                "AnnealCalibrator requires the viennaps Python bindings. "
                "Build ViennaPS with -DVIENNAPS_BUILD_PYTHON=ON."
            )

        self._vps  = vps
        self._core = _core
        self._n_eval = 0

        if self._fit_ss:
            param_keys = ["log10_D0", "annealEa", "log10_C0_ss", "Ea_ss"]
        else:
            param_keys = ["log10_D0", "annealEa"]
        bvec = [merged[k] for k in param_keys]

        obs = self.post_anneal.intensity

        def objective(x: np.ndarray) -> float:
            self._n_eval += 1
            args = dict(zip(param_keys, x))
            pred, rsh_sim, _ = self._run_simulation(**args)
            if pred is None:
                return 1e6
            return self._combined_loss_with_rsh(pred, obs, rsh_sim)

        best_x = run_optimizer(objective, bvec, n_budget, seed, optimizer,
                               polish=False)
        best   = dict(zip(param_keys, best_x))

        # Final evaluation to collect profile and Rsh
        pred, rsh_sim, (sim_d, sim_v) = self._run_simulation(**best)
        if pred is None:
            raise RuntimeError("Final anneal simulation produced no dopant profile.")
        pred_clip = np.maximum(pred, self.floor)

        rsh_err_pct = None
        if rsh_sim is not None and self.rsh_target is not None:
            rsh_err_pct = 100.0 * (rsh_sim - self.rsh_target) / self.rsh_target

        # Normalise stored sim profile for plotting
        sim_norm = None
        if sim_v is not None and sim_v.max() > 0:
            sim_norm = sim_v / sim_v.max()

        return AnnealFitResult(
            annealD0      = float(10.0 ** best["log10_D0"]),
            annealEa      = float(best["annealEa"]),
            solidSolC0    = float(10.0 ** best["log10_C0_ss"]) if self._fit_ss else None,
            solidSolEa    = float(best["Ea_ss"])               if self._fit_ss else None,
            sim_depths    = sim_d,
            sim_profile   = sim_norm,
            r2_log        = r2_log(pred_clip, obs, self.floor),
            rmse_log      = rmse_log(pred_clip, obs, self.floor),
            rsh_computed  = rsh_sim,
            rsh_target    = self.rsh_target,
            rsh_error_pct = rsh_err_pct,
            n_eval        = self._n_eval,
            optimizer     = f"anneal+{optimizer}",
        )

    # ── simulation core ───────────────────────────────────────────────────────

    def _run_simulation(
        self,
        log10_D0:    float,
        annealEa:    float,
        log10_C0_ss: Optional[float] = None,
        Ea_ss:       Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[float],
               Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Build domain, initialise from SIMS, run Anneal, extract profile + Rsh.

        Returns
        -------
        pred        : normalised profile interpolated on post-anneal SIMS depths
        rsh_sim     : sheet resistance [Ω/□], or None if solid activation off
        (depths, vals): raw extracted profile (nm, nm⁻³) for storage / plotting
        """
        vps  = self._vps
        core = self._core

        domain = self.domain_factory()

        # Initialise cell set with pre-anneal SIMS profile (nm⁻³ units)
        init_domain_from_sims(
            domain, self.species,
            self.pre_anneal.depth, self.pre_anneal.intensity,
            dose_cm2=self.dose_cm2,
        )

        # Configure anneal
        fit_ss = (self._fit_ss
                  and log10_C0_ss is not None
                  and Ea_ss is not None)

        anneal = vps.Anneal()
        anneal.setTemperature(self.temp_K)
        anneal.setDuration(self.time_s)
        anneal.setArrheniusParameters(10.0 ** log10_D0, annealEa)
        anneal.setSpeciesLabel(self._label_total)
        anneal.setDiffusionMaterials([core.Material.Si])
        anneal.setBlockingMaterials([core.Material.Air])

        if fit_ss:
            anneal.enableSolidActivation(True)
            anneal.setSolidSolubilityArrhenius(10.0 ** log10_C0_ss, Ea_ss)
            anneal.setActiveLabel(self._label_active)

        vps.Process(domain, anneal, 0.).apply()

        # Extract post-anneal profile
        extract_label = self._label_active if fit_ss else self._label_total
        sim_depths, sim_vals = extract_depth_profile(
            domain, extract_label, depth_axis=1, agg=self.depth_agg)

        if len(sim_depths) == 0 or sim_vals.max() <= 0:
            return None, None, (None, None)

        # Compute Rsh via ViennaCS SheetResistance (Masetti electron model)
        rsh_sim = None
        if fit_ss:
            sr = vps.SheetResistance()
            sr.setCellSet(domain.getCellSet())
            sr.setConcentrationLabel(self._label_active)
            rsh_sim = float(sr.computeElectron())

        # Normalise and interpolate onto post-anneal SIMS depth axis
        peak = sim_vals.max()
        sim_norm = sim_vals / peak
        pred = np.interp(
            self.post_anneal.depth, sim_depths, sim_norm,
            left=0., right=0.)

        return pred, rsh_sim, (sim_depths, sim_vals)

    # ── loss functions ────────────────────────────────────────────────────────

    def _combined_loss_with_rsh(self, pred: np.ndarray, obs: np.ndarray,
                                 rsh_sim: Optional[float]) -> float:
        profile_loss = residual_loss(pred, obs, floor=self.floor,
                                     log_fit=self.log_fit, reduction="mean")

        if (self.rsh_weight <= 0.0 or rsh_sim is None
                or self.rsh_target is None):
            return profile_loss

        rsh_log_err = (np.log10(max(rsh_sim, 1.0))
                       - np.log10(max(self.rsh_target, 1.0))) ** 2
        return ((1.0 - self.rsh_weight) * profile_loss
                + self.rsh_weight * rsh_log_err)
