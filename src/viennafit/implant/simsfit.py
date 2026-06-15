"""
Calibrate Pearson IV (single or dual) implant parameters to SIMS depth profiles.

Fitting is performed in log-space so all decades of the profile contribute
equally, which is appropriate for SIMS data that spans 2-4 orders of magnitude.

Supported global-search backends (``optimizer=`` argument):
  "scipy"     — scipy differential_evolution + L-BFGS-B refinement (default,
                no extra dependencies beyond NumPy/SciPy)
  "dlib"      — dlib.find_min_global (Gaussian-process surrogate; same library
                used by the rest of ViennaFit)
  "nevergrad" — Nevergrad DE strategy (same library used by ViennaFit)
  "cma"       — CMA-ES via the cma package (same library used by ViennaFit)

All backends are optional; a helpful ImportError is raised if the requested
library is not installed.

Typical usage
-------------
from viennafit.implant import SimsProfile, PearsonIVFitter

profile = SimsProfile.from_csv("sims_5keV_P_Si.csv")

# Dual Pearson IV with the default scipy backend
result = PearsonIVFitter(profile, dual=True).fit()

# Same fit using the dlib optimizer (if ViennaFit dependencies are installed)
result = PearsonIVFitter(profile, dual=True).fit(optimizer="dlib")

result.print_summary()
result.plot(profile)
print(result.to_config_string())
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._common import (
    OPTIMIZERS,
    Optimizer,
    residual_loss,
    r2_log,
    rmse_log,
    run_optimizer,
)
from .pearsoniv import dual_pearson_profile, pearson_profile


# ─── Data container ──────────────────────────────────────────────────────────

class SimsProfile:
    """
    Holds a 1-D SIMS depth profile.

    Attributes
    ----------
    depth     : 1-D array of depths [nm], monotonically increasing, ≥ 0
    intensity : 1-D array of SIMS signal / concentration (any consistent units);
                values must be > 0 (log-space fitting requirement)
    label     : short description used in plot legends
    """

    def __init__(self, depth: np.ndarray, intensity: np.ndarray,
                 label: str = "SIMS"):
        depth = np.asarray(depth, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if depth.shape != intensity.shape:
            raise ValueError("depth and intensity must have the same length")
        mask = (depth >= 0) & (intensity > 0)
        self.depth     = depth[mask]
        self.intensity = intensity[mask]
        self.label     = label

    @classmethod
    def from_csv(cls, path: str | Path, depth_col: str = "depth_nm",
                 intensity_col: str = "intensity",
                 label: Optional[str] = None) -> "SimsProfile":
        """
        Load SIMS data from a CSV file.

        Expected columns: ``depth_nm`` and ``intensity`` (names configurable).
        Lines starting with ``#`` are treated as comments.
        """
        path = Path(path)
        depth_vals: List[float] = []
        intens_vals: List[float] = []
        header: Optional[List[str]] = None

        with open(path, newline="") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if header is None:
                    header = [c.strip() for c in line.split(",")]
                    continue
                parts = [c.strip() for c in line.split(",")]
                row = dict(zip(header, parts))
                try:
                    depth_vals.append(float(row[depth_col]))
                    intens_vals.append(float(row[intensity_col]))
                except KeyError:
                    raise ValueError(
                        f"CSV must contain columns '{depth_col}' and "
                        f"'{intensity_col}'. Found: {header}"
                    )

        lbl = label or path.stem
        return cls(np.array(depth_vals), np.array(intens_vals), label=lbl)

    def normalize(self) -> "SimsProfile":
        """Return a copy with intensity normalized so that the peak equals 1."""
        return SimsProfile(self.depth.copy(),
                           self.intensity / self.intensity.max(),
                           label=self.label)

    def __len__(self) -> int:
        return len(self.depth)


# ─── Fit result ───────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """
    Fitted Pearson IV parameters and diagnostics.

    All parameter names match the ViennaPS config.txt keys so they can be
    copied directly into a simulation config file.
    """
    # Head component
    projectedRange : float
    depthSigma     : float
    skewness       : float   # → C++ params.beta  (β₂ position in Pearson formula)
    kurtosis       : float   # → C++ params.gamma (γ₁ position in Pearson formula)

    # Tail component (set only for dual fits)
    headFraction          : Optional[float] = None
    tailProjectedRange    : Optional[float] = None
    tailDepthSigma        : Optional[float] = None
    tailSkewness          : Optional[float] = None
    tailKurtosis          : Optional[float] = None

    # Dose scale (not written to config)
    amplitude  : float = 1.0

    # Diagnostics
    r2_log     : float = float("nan")
    rmse_log   : float = float("nan")
    n_eval     : int   = 0
    dual       : bool  = False
    optimizer  : str   = "scipy"

    def to_config_string(self, prefix: str = "") -> str:
        """
        Return the fitted parameters as ViennaPS config.txt lines.

        Parameters
        ----------
        prefix : optional prefix for each line (e.g. "# " to comment them out)
        """
        lines = [
            f"projectedRange={self.projectedRange:.6g}",
            f"depthSigma={self.depthSigma:.6g}",
            f"skewness={self.skewness:.6g}",
            f"kurtosis={self.kurtosis:.6g}",
        ]
        if self.dual and self.headFraction is not None:
            lines += [
                f"headFraction={self.headFraction:.6g}",
                f"tailProjectedRange={self.tailProjectedRange:.6g}",
                f"tailDepthSigma={self.tailDepthSigma:.6g}",
                f"tailSkewness={self.tailSkewness:.6g}",
                f"tailKurtosis={self.tailKurtosis:.6g}",
            ]
        return "\n".join(prefix + ln for ln in lines)

    def print_summary(self) -> None:
        """Print a formatted summary of the fit result."""
        print("=" * 52)
        print(f"  Pearson IV fit  [{self.optimizer}]")
        print("=" * 52)
        print(f"  projectedRange : {self.projectedRange:.4f} nm")
        print(f"  depthSigma     : {self.depthSigma:.4f} nm")
        print(f"  skewness       : {self.skewness:.4f}  (→ C++ params.beta)")
        print(f"  kurtosis       : {self.kurtosis:.4f}  (→ C++ params.gamma)")
        if self.dual and self.headFraction is not None:
            print(f"  headFraction         : {self.headFraction:.4f}")
            print(f"  tailProjectedRange   : {self.tailProjectedRange:.4f} nm")
            print(f"  tailDepthSigma       : {self.tailDepthSigma:.4f} nm")
            print(f"  tailSkewness         : {self.tailSkewness:.4f}")
            print(f"  tailKurtosis         : {self.tailKurtosis:.4f}")
        print("-" * 52)
        print(f"  R² (log-space) : {self.r2_log:.4f}")
        print(f"  RMSE (log)     : {self.rmse_log:.4f}")
        print(f"  evaluations    : {self.n_eval}")
        print("=" * 52)
        print()
        print("Config.txt block:")
        print(textwrap.indent(self.to_config_string(), "  "))

    def plot(self, sims_profile: Optional[SimsProfile] = None,
             ax=None, show: bool = True) -> None:
        """
        Plot the fit against the SIMS data.

        Parameters
        ----------
        sims_profile : original SimsProfile for data overlay
        ax           : matplotlib Axes; a new figure is created if None
        show         : call plt.show() after drawing
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        if sims_profile is not None:
            ax.semilogy(sims_profile.depth, sims_profile.intensity,
                        "o", color="C0", ms=5, label=f"SIMS ({sims_profile.label})")

        z_max = sims_profile.depth.max() if sims_profile else self.projectedRange * 5
        z_grid = np.linspace(0, z_max, 500)

        if self.dual and self.headFraction is not None:
            total = dual_pearson_profile(
                z_grid,
                self.projectedRange, self.depthSigma, self.skewness, self.kurtosis,
                self.tailProjectedRange, self.tailDepthSigma,
                self.tailSkewness, self.tailKurtosis,
                self.headFraction,
            )
            head = pearson_profile(z_grid, self.projectedRange, self.depthSigma,
                                   self.skewness, self.kurtosis)
            tail = pearson_profile(z_grid, self.tailProjectedRange, self.tailDepthSigma,
                                   self.tailSkewness, self.tailKurtosis)
            ax.semilogy(z_grid, self.amplitude * total,
                        "-", color="C1", lw=2, label="Dual-Pearson IV fit")
            ax.semilogy(z_grid, self.amplitude * self.headFraction * head,
                        "--", color="C2", lw=1.2, alpha=0.8, label="Head component")
            ax.semilogy(z_grid, self.amplitude * (1 - self.headFraction) * tail,
                        "--", color="C3", lw=1.2, alpha=0.8, label="Tail component")
        else:
            profile = pearson_profile(z_grid, self.projectedRange, self.depthSigma,
                                      self.skewness, self.kurtosis)
            ax.semilogy(z_grid, self.amplitude * profile,
                        "-", color="C1", lw=2, label="Pearson IV fit")

        ax.set_xlabel("Depth (nm)")
        ax.set_ylabel("Concentration (a.u.)")
        ax.set_title(f"Pearson IV fit to SIMS data  [optimizer: {self.optimizer}]")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()


# ─── Default parameter bounds ────────────────────────────────────────────────

_DEFAULT_BOUNDS_SINGLE: Dict[str, Tuple[float, float]] = {
    "projectedRange": (0.5,  50.0),
    "depthSigma":     (0.3,  25.0),
    "skewness":       (0.5,  30.0),   # → params.beta (β₂ position in formula)
    "kurtosis":       (-3.0,  5.0),   # → params.gamma (γ₁ position in formula)
    "log_amplitude":  (-5.0,  5.0),
}

_DEFAULT_BOUNDS_TAIL: Dict[str, Tuple[float, float]] = {
    "tailProjectedRange": (5.0,  200.0),
    "tailDepthSigma":     (2.0,   80.0),
    "tailSkewness":       (0.5,   15.0),
    "tailKurtosis":       (-2.0,   4.0),
    "headFraction":       (0.5,   0.999),
}


# ─── Fitter ───────────────────────────────────────────────────────────────────

class PearsonIVFitter:
    """
    Fit a single or dual Pearson IV distribution to a SIMS depth profile.

    Parameters
    ----------
    profile   : SimsProfile  —  experimental data
    dual      : fit dual Pearson IV (head + tail) when True
    log_fit   : minimize residuals in log-space (recommended, default True)
    min_log10 : floor for log₁₀(intensity) to guard against zeros (default -6)

    Example
    -------
    fitter = PearsonIVFitter(profile, dual=True)
    result = fitter.fit(optimizer="scipy")   # or "dlib", "nevergrad", "cma"
    result.print_summary()
    result.plot(profile)
    print(result.to_config_string())
    """

    def __init__(self, profile: SimsProfile, *, dual: bool = False,
                 log_fit: bool = True, min_log10: float = -6.0):
        self.profile = profile.normalize()
        self.dual    = dual
        self.log_fit = log_fit
        self.floor   = 10.0 ** min_log10
        self._n_eval = 0

    def fit(self,
            head_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            tail_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            n_budget: int = 1200,
            seed: int = 42,
            optimizer: Optimizer = "scipy") -> FitResult:
        """
        Fit the profile.

        Parameters
        ----------
        head_bounds : override default bounds for head (and single-fit) parameters
        tail_bounds : override default bounds for tail parameters (dual only)
        n_budget    : total objective evaluations budget for the global search
        seed        : random seed for reproducibility
        optimizer   : "scipy" (default) | "dlib" | "nevergrad" | "cma"

        Returns
        -------
        FitResult with fitted parameters and diagnostics.
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}; "
                             f"got '{optimizer}'")
        hb = {**_DEFAULT_BOUNDS_SINGLE, **(head_bounds or {})}
        tb = {**_DEFAULT_BOUNDS_TAIL,   **(tail_bounds or {})}

        if self.dual:
            return self._fit_dual(hb, tb, n_budget, seed, optimizer)
        else:
            return self._fit_single(hb, n_budget, seed, optimizer)

    # ── single Pearson IV ─────────────────────────────────────────────────────

    def _fit_single(self, bounds: dict, n_budget: int, seed: int,
                    optimizer: str) -> FitResult:
        keys  = ["projectedRange", "depthSigma", "skewness", "kurtosis",
                 "log_amplitude"]
        bvec  = [bounds[k] for k in keys]
        depth = self.profile.depth
        obs   = self.profile.intensity
        self._n_eval = 0

        def objective(x):
            self._n_eval += 1
            rp, dRp, beta, gamma, log_amp = x
            amp  = 10.0 ** log_amp
            pred = amp * pearson_profile(depth, rp, dRp, beta, gamma)
            return residual_loss(pred, obs, floor=self.floor,
                                 log_fit=self.log_fit, reduction="sum")

        best = run_optimizer(objective, bvec, n_budget, seed, optimizer,
                             polish=True)
        rp, dRp, beta, gamma, log_amp = best
        amp       = 10.0 ** log_amp
        pred_best = amp * pearson_profile(depth, rp, dRp, beta, gamma)

        return FitResult(
            projectedRange=float(rp), depthSigma=float(dRp),
            skewness=float(beta), kurtosis=float(gamma),
            amplitude=float(amp),
            r2_log=r2_log(pred_best, obs, self.floor),
            rmse_log=rmse_log(pred_best, obs, self.floor),
            n_eval=self._n_eval, dual=False, optimizer=optimizer,
        )

    # ── dual Pearson IV ───────────────────────────────────────────────────────

    def _fit_dual(self, hb: dict, tb: dict, n_budget: int, seed: int,
                  optimizer: str) -> FitResult:
        head_keys = ["projectedRange", "depthSigma", "skewness", "kurtosis",
                     "log_amplitude"]
        tail_keys = ["tailProjectedRange", "tailDepthSigma",
                     "tailSkewness", "tailKurtosis", "headFraction"]
        all_bounds = [hb[k] for k in head_keys] + [tb[k] for k in tail_keys]
        depth = self.profile.depth
        obs   = self.profile.intensity
        self._n_eval = 0

        def objective(x):
            self._n_eval += 1
            rp, dRp, beta, gamma, log_amp = x[:5]
            rp_t, dRp_t, beta_t, gamma_t, hf = x[5:]
            amp  = 10.0 ** log_amp
            pred = amp * dual_pearson_profile(
                depth,
                rp, dRp, beta, gamma,
                rp_t, dRp_t, beta_t, gamma_t,
                hf,
            )
            return residual_loss(pred, obs, floor=self.floor,
                                 log_fit=self.log_fit, reduction="sum")

        best = run_optimizer(objective, all_bounds, n_budget, seed, optimizer,
                             polish=True)
        rp, dRp, beta, gamma, log_amp = best[:5]
        rp_t, dRp_t, beta_t, gamma_t, hf = best[5:]
        amp       = 10.0 ** log_amp
        pred_best = amp * dual_pearson_profile(
            depth, rp, dRp, beta, gamma,
            rp_t, dRp_t, beta_t, gamma_t, hf)

        return FitResult(
            projectedRange=float(rp), depthSigma=float(dRp),
            skewness=float(beta), kurtosis=float(gamma),
            headFraction=float(np.clip(hf, 0.0, 1.0)),
            tailProjectedRange=float(rp_t), tailDepthSigma=float(dRp_t),
            tailSkewness=float(beta_t), tailKurtosis=float(gamma_t),
            amplitude=float(amp),
            r2_log=r2_log(pred_best, obs, self.floor),
            rmse_log=rmse_log(pred_best, obs, self.floor),
            n_eval=self._n_eval, dual=True, optimizer=optimizer,
        )
