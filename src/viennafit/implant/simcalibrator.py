"""
Simulation-in-the-loop Pearson IV calibration.

``SimulationCalibrator`` runs the full ViennaPS implant (and optionally anneal)
simulation inside each optimizer call, then compares the extracted depth profile
against SIMS data.  It reuses the same pluggable optimizer backends as
``PearsonIVFitter`` (scipy / dlib / nevergrad / cma).

Typical two-step workflow
-------------------------
Step 1 — fast analytical pre-fit (no ViennaPS needed):

    from viennafit.implant import SimsProfile, PearsonIVFitter
    profile = SimsProfile.from_csv("sims_5keV_P_Si.csv")
    guess = PearsonIVFitter(profile, dual=True).fit()

Step 2 — simulation refinement (uses ViennaPS Python bindings):

    from viennafit.implant import SimulationCalibrator, build_masked_substrate_domain
    import functools

    domain_factory = functools.partial(
        build_masked_substrate_domain,
        grid_delta=1.0, x_extent=200., top_space=15.,
        substrate_depth=80., opening_width=100.,
        mask_height=20., oxide_thickness=2.,
    )

    cal = SimulationCalibrator(
        sims_profile   = profile,
        domain_factory = domain_factory,
        species        = "P",
        dose_cm2       = 1e14,
        tilt_angle     = 6.,
        screen_thickness = 2.,
        dual           = True,
    )
    result = cal.fit(initial_guess=guess, n_budget=200, optimizer="scipy")
    result.print_summary()
    print(result.to_config_string())
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ._common import OPTIMIZERS, residual_loss, r2_log, rmse_log, run_optimizer
from .simsfit import (
    FitResult,
    SimsProfile,
    _DEFAULT_BOUNDS_SINGLE,
    _DEFAULT_BOUNDS_TAIL,
)
from .simprofile import extract_depth_profile

class SimulationCalibrator:
    """
    Calibrate Pearson IV implant moments against SIMS data by running the full
    ViennaPS simulation in each optimizer iteration.

    Parameters
    ----------
    sims_profile     : SimsProfile — the experimental SIMS target
    domain_factory   : callable ``() → domain`` that builds a fresh ViennaPS
                       2-D domain with the cell set already generated.
                       Use ``build_masked_substrate_domain(...)`` or provide
                       your own factory.  Called once per optimizer evaluation.
    species          : dopant species label (e.g. ``"P"``); determines cell-set
                       field names ``{species}_total`` etc.
    dose_cm2         : implant dose in ions/cm²
    tilt_angle       : beam tilt in degrees
    screen_thickness : screen-oxide thickness subtracted from the Pearson IV mean
                       (same value as ``screenThickness`` in config.txt) [nm]
    dual             : if True, optimise dual Pearson IV (head + tail)
    log_fit          : minimise residuals in log-space (recommended)
    min_log10        : intensity floor for log conversion (default −6)
    depth_agg        : ``"max"`` (default) — compare peak concentration across x;
                       ``"sum"`` — compare laterally integrated dose per slice
    """

    def __init__(
        self,
        sims_profile: SimsProfile,
        domain_factory: Callable,
        *,
        species: str = "P",
        dose_cm2: float = 1e14,
        tilt_angle: float = 7.,
        screen_thickness: float = 0.,
        dual: bool = True,
        log_fit: bool = True,
        min_log10: float = -6.,
        depth_agg: str = "max",
    ):
        self.profile          = sims_profile.normalize()
        self.domain_factory   = domain_factory
        self.species          = species
        self.dose_cm2         = dose_cm2
        self.tilt_angle       = tilt_angle
        self.screen_thickness = screen_thickness
        self.dual             = dual
        self.log_fit          = log_fit
        self.floor            = 10.0 ** min_log10
        self.depth_agg        = depth_agg
        self._n_eval          = 0

        self._label_total = f"{species}_total"
        self._label_damage = f"{species}_damage"

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        initial_guess: Optional[FitResult] = None,
        head_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        tail_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_budget: int = 200,
        optimizer: str = "scipy",
        seed: int = 42,
    ) -> FitResult:
        """
        Run the calibration.

        Parameters
        ----------
        initial_guess : FitResult from a prior ``PearsonIVFitter.fit()`` call.
                        When provided, the DE starting population is seeded with
                        this point, which can dramatically speed up convergence.
        head_bounds   : override default bounds for head parameters
        tail_bounds   : override default bounds for tail parameters (dual only)
        n_budget      : total simulation calls budget (keep ≤ 500 for speed)
        optimizer     : ``"scipy"`` | ``"dlib"`` | ``"nevergrad"`` | ``"cma"``
        seed          : random seed

        Returns
        -------
        FitResult — same type as PearsonIVFitter, so
        ``result.to_config_string()`` and ``result.plot()`` work unchanged.
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}")

        hb = {**_DEFAULT_BOUNDS_SINGLE, **(head_bounds or {})}
        tb = {**_DEFAULT_BOUNDS_TAIL,   **(tail_bounds or {})}

        try:
            import viennaps.d2 as vps
            import viennaps as _core
        except ImportError:
            raise ImportError(
                "SimulationCalibrator requires viennaps Python bindings. "
                "Build ViennaPS with -DVIENNAPS_BUILD_PYTHON=ON."
            )

        self._vps  = vps
        self._core = _core
        self._n_eval = 0

        if self.dual:
            return self._fit_dual(hb, tb, n_budget, seed, optimizer,
                                  initial_guess)
        else:
            return self._fit_single(hb, n_budget, seed, optimizer,
                                    initial_guess)

    # ── simulation objective ──────────────────────────────────────────────────

    def _run_simulation(self, rp, dRp, beta, gamma, log_amp,
                        rp_t=None, dRp_t=None, beta_t=None,
                        gamma_t=None, hf=None):
        """Build domain, run implant, return interpolated simulated profile."""
        vps  = self._vps
        core = self._core

        domain = self.domain_factory()

        # Pearson IV head parameters
        head = core.PearsonIVParameters()
        head.mu    = rp - self.screen_thickness
        head.sigma = dRp
        head.beta  = beta
        head.gamma = gamma

        if self.dual and rp_t is not None:
            tail = core.PearsonIVParameters()
            tail.mu    = rp_t - self.screen_thickness
            tail.sigma = dRp_t
            tail.beta  = beta_t
            tail.gamma = gamma_t
            # Use zero lateral spread defaults — lateral shape not
            # resolved in a 1-D profile comparison
            implant_model = vps.ImplantDualPearsonIV(
                head, tail, float(np.clip(hf, 0., 1.)),
                0., max(dRp * 0.3, 1.),   # head lateral spread
                0., max(dRp_t * 0.3, 1.),  # tail lateral spread
            )
        else:
            implant_model = vps.ImplantPearsonIV(head, 0., max(dRp * 0.3, 1.))

        implant = vps.IonImplantation()
        implant.setImplantModel(implant_model)
        implant.setDose(self.dose_cm2)
        implant.setTiltAngle(self.tilt_angle)
        implant.setLengthUnit(1e-7)
        implant.setDoseControl(core.ImplantDoseControl.WaferDose)
        implant.setMaskMaterials([core.Material.Mask])
        implant.setScreenMaterials([core.Material.SiO2])
        implant.setConcentrationLabel(self._label_total)
        implant.setDamageLabel(self._label_damage)

        vps.Process(domain, implant, 0.).apply()

        # Extract depth profile and interpolate onto SIMS depth axis
        sim_depths, sim_vals = extract_depth_profile(
            domain, self._label_total, depth_axis=1, agg=self.depth_agg)

        if len(sim_depths) == 0 or sim_vals.max() <= 0:
            return None

        sim_vals = sim_vals / max(sim_vals.max(), 1e-30)  # normalize to peak=1

        # Interpolate simulated profile onto SIMS measurement depths
        interp = np.interp(self.profile.depth, sim_depths, sim_vals,
                           left=0., right=0.)
        return (10.0 ** log_amp) * interp

    # ── single Pearson IV ─────────────────────────────────────────────────────

    def _fit_single(self, hb, n_budget, seed, optimizer, guess):
        keys  = ["projectedRange", "depthSigma", "skewness", "kurtosis",
                 "log_amplitude"]
        bvec  = [hb[k] for k in keys]
        obs   = self.profile.intensity

        x0 = self._guess_to_vec_single(guess, hb) if guess else None

        def objective(x):
            self._n_eval += 1
            rp, dRp, beta, gamma, log_amp = x
            pred = self._run_simulation(rp, dRp, beta, gamma, log_amp)
            if pred is None:
                return 1e6
            return residual_loss(pred, obs, floor=self.floor,
                                 log_fit=self.log_fit, reduction="sum")

        best = run_optimizer(objective, bvec, n_budget, seed, optimizer,
                             initial_guess=x0, polish=False)
        rp, dRp, beta, gamma, log_amp = best
        amp  = 10.0 ** log_amp
        pred = self._run_simulation(rp, dRp, beta, gamma, 0.)  # normalized
        if pred is None:
            raise RuntimeError("Final simulation produced no implant profile.")
        pred_scaled = pred * amp

        return FitResult(
            projectedRange=float(rp), depthSigma=float(dRp),
            skewness=float(beta), kurtosis=float(gamma),
            amplitude=float(amp),
            r2_log=r2_log(pred_scaled, obs, self.floor),
            rmse_log=rmse_log(pred_scaled, obs, self.floor),
            n_eval=self._n_eval, dual=False, optimizer=f"sim+{optimizer}",
        )

    # ── dual Pearson IV ───────────────────────────────────────────────────────

    def _fit_dual(self, hb, tb, n_budget, seed, optimizer, guess):
        head_keys = ["projectedRange", "depthSigma", "skewness", "kurtosis",
                     "log_amplitude"]
        tail_keys = ["tailProjectedRange", "tailDepthSigma",
                     "tailSkewness", "tailKurtosis", "headFraction"]
        all_bounds = [hb[k] for k in head_keys] + [tb[k] for k in tail_keys]
        obs = self.profile.intensity
        x0 = self._guess_to_vec_dual(guess, hb, tb) if guess else None

        def objective(x):
            self._n_eval += 1
            rp, dRp, beta, gamma, log_amp = x[:5]
            rp_t, dRp_t, beta_t, gamma_t, hf = x[5:]
            pred = self._run_simulation(rp, dRp, beta, gamma, log_amp,
                                        rp_t, dRp_t, beta_t, gamma_t, hf)
            if pred is None:
                return 1e6
            return residual_loss(pred, obs, floor=self.floor,
                                 log_fit=self.log_fit, reduction="sum")

        best = run_optimizer(objective, all_bounds, n_budget, seed, optimizer,
                             initial_guess=x0, polish=False)
        rp, dRp, beta, gamma, log_amp = best[:5]
        rp_t, dRp_t, beta_t, gamma_t, hf = best[5:]
        amp  = 10.0 ** log_amp
        pred = self._run_simulation(rp, dRp, beta, gamma, 0.,
                                    rp_t, dRp_t, beta_t, gamma_t, hf)
        if pred is None:
            raise RuntimeError("Final simulation produced no implant profile.")
        pred_scaled = pred * amp

        return FitResult(
            projectedRange=float(rp), depthSigma=float(dRp),
            skewness=float(beta), kurtosis=float(gamma),
            headFraction=float(np.clip(hf, 0., 1.)),
            tailProjectedRange=float(rp_t), tailDepthSigma=float(dRp_t),
            tailSkewness=float(beta_t), tailKurtosis=float(gamma_t),
            amplitude=float(amp),
            r2_log=r2_log(pred_scaled, obs, self.floor),
            rmse_log=rmse_log(pred_scaled, obs, self.floor),
            n_eval=self._n_eval, dual=True, optimizer=f"sim+{optimizer}",
        )

    # ── initial guess helpers ─────────────────────────────────────────────────

    @staticmethod
    def _guess_to_vec_single(guess: FitResult, hb: dict) -> list:
        lo = np.array([hb[k][0] for k in
                       ["projectedRange", "depthSigma", "skewness",
                        "kurtosis", "log_amplitude"]])
        hi = np.array([hb[k][1] for k in
                       ["projectedRange", "depthSigma", "skewness",
                        "kurtosis", "log_amplitude"]])
        x = np.array([guess.projectedRange, guess.depthSigma,
                      guess.skewness, guess.kurtosis,
                      np.log10(max(guess.amplitude, 1e-10))])
        return np.clip(x, lo, hi).tolist()

    @staticmethod
    def _guess_to_vec_dual(guess: FitResult, hb: dict, tb: dict) -> list:
        head_keys = ["projectedRange", "depthSigma", "skewness", "kurtosis",
                     "log_amplitude"]
        tail_keys = ["tailProjectedRange", "tailDepthSigma",
                     "tailSkewness", "tailKurtosis", "headFraction"]
        lo = np.array([hb[k][0] for k in head_keys] +
                      [tb[k][0] for k in tail_keys])
        hi = np.array([hb[k][1] for k in head_keys] +
                      [tb[k][1] for k in tail_keys])
        x = np.array([
            guess.projectedRange,
            guess.depthSigma,
            guess.skewness,
            guess.kurtosis,
            np.log10(max(guess.amplitude, 1e-10)),
            guess.tailProjectedRange if guess.tailProjectedRange is not None
            else tb["tailProjectedRange"][0],
            guess.tailDepthSigma if guess.tailDepthSigma is not None
            else tb["tailDepthSigma"][0],
            guess.tailSkewness if guess.tailSkewness is not None
            else tb["tailSkewness"][0],
            guess.tailKurtosis if guess.tailKurtosis is not None
            else tb["tailKurtosis"][0],
            guess.headFraction if guess.headFraction is not None else 0.9,
        ])
        return np.clip(x, lo, hi).tolist()
