"""
Pearson distribution for ion implant depth profiles.

Direct Python/NumPy port of ViennaPS psImplantConstants.hpp::PearsonIV.
All operations are vectorized over depth arrays for speed in optimization loops.

Parameter convention (matches ViennaPS config.txt keys):
  mu        -- projected range / mean depth  [length units, e.g. nm]
  sigma     -- straggle / standard deviation  [same units]
  skewness  -- maps to C++ params.beta;
               appears in the β₂ (kurtosis) position of the Pearson formula
  kurtosis  -- maps to C++ params.gamma;
               appears in the γ₁ (signed skewness) position of the formula

The function automatically selects the correct Pearson branch per-element:
  discriminant ≤ 0  →  Type I/II  (bounded support, atanh log-space)
  discriminant > 0  →  Type IV    (arctan log-space)
"""

from __future__ import annotations

import numpy as np


# ─── Vectorized Pearson kernel ────────────────────────────────────────────────

def pearson_unnorm(z: np.ndarray | float,
                   mu: float, sigma: float,
                   skewness: float, kurtosis: float) -> np.ndarray:
    """
    Unnormalized Pearson profile value at depth(s) z.

    Vectorized: z may be a scalar or any numpy array.
    Values at z < 0 are 0 (surface boundary).
    Returns 0 for numerically invalid combinations.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)

    if sigma <= 0.0:
        return out

    beta  = float(skewness)  # C++ params.beta
    gamma = float(kurtosis)  # C++ params.gamma

    A = 10.0 * beta - 12.0 * gamma * gamma - 18.0
    if abs(A) < 1e-12:
        return out

    a  = -gamma * sigma * (beta + 3.0) / A
    b0 = -sigma * sigma * (4.0 * beta - 3.0 * gamma * gamma) / A
    b1 = a
    b2 = -(2.0 * beta - 3.0 * gamma * gamma - 6.0) / A

    if abs(b2) < 1e-12:
        return out

    discriminant = 4.0 * b0 * b2 - b1 * b1
    m = 1.0 / (2.0 * b2)

    # Shift to mean and mask negative depths
    valid = z >= 0.0
    x = np.where(valid, z - mu, 0.0)
    poly = b0 + b1 * x + b2 * x * x

    with np.errstate(all="ignore"):
        if discriminant <= 0.0:
            # ── Type I / II  (atanh branch) ────────────────────────────────
            sqrt_neg = np.sqrt(max(-discriminant, 0.0))
            if sqrt_neg < 1e-30:
                return out
            arg = (2.0 * b2 * x + b1) / sqrt_neg
            abs_poly = np.abs(poly)
            log_ok = valid & (np.abs(arg) < 1.0) & (abs_poly > 0.0)
            log_r = np.where(log_ok,
                             m * np.log(np.where(abs_poly > 0, abs_poly, 1.0))
                             + (b1 / b2 + 2.0 * a) / sqrt_neg
                             * np.arctanh(np.clip(arg, -1.0 + 1e-15, 1.0 - 1e-15)),
                             -np.inf)
            finite_ok = log_ok & np.isfinite(log_r) & (log_r < 700.0)
            out = np.where(finite_ok, np.exp(np.where(finite_ok, log_r, 0.0)), 0.0)

        else:
            # ── Type IV  (arctan branch) ──────────────────────────────────
            sqrt_disc = np.sqrt(discriminant)
            abs_poly = np.abs(poly)
            exponent = (-(b1 / b2 + 2.0 * a) / sqrt_disc
                        * np.arctan((2.0 * b2 * x + b1) / sqrt_disc))
            ok = valid & (abs_poly > 0.0) & np.isfinite(exponent) & (exponent < 700.0)
            base = np.where(ok, abs_poly, 1.0)
            result = np.where(ok,
                              np.exp(m * np.log(base) + np.where(ok, exponent, 0.0)),
                              0.0)
            out = np.where(np.isfinite(result), result, 0.0)

    return out


# ─── Normalization (trapezoidal, matches C++ integrateTrapezoidal) ────────────

def _normalization(mu: float, sigma: float,
                   skewness: float, kurtosis: float,
                   z_max: float | None = None) -> float:
    """Integral of unnormalized profile from 0 to z_max (trapezoidal rule)."""
    if z_max is None:
        z_max = max(mu + 8.0 * sigma, 1.0)
    step = max(sigma / 50.0, 1e-3)
    z_grid = np.arange(0.0, z_max + step * 0.5, step)
    with np.errstate(all="ignore"):
        vals = pearson_unnorm(z_grid, mu, sigma, skewness, kurtosis)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    val = float(np.trapezoid(vals, z_grid))
    return val if val > 0.0 else 1.0


# ─── Normalized profiles ──────────────────────────────────────────────────────

def pearson_profile(z: np.ndarray | float,
                    mu: float, sigma: float,
                    skewness: float, kurtosis: float,
                    z_max: float | None = None) -> np.ndarray:
    """
    Normalized Pearson depth profile (integrates to 1 over [0, z_max]).

    Parameters
    ----------
    z        : depth(s) in nm (negative returns 0)
    mu       : projected range [nm]  (config key: projectedRange)
    sigma    : straggle [nm]          (config key: depthSigma)
    skewness : → C++ params.beta     (config key: skewness)
    kurtosis : → C++ params.gamma    (config key: kurtosis)
    z_max    : upper integration limit for normalization [nm]; default mu+8σ
    """
    raw  = pearson_unnorm(z, mu, sigma, skewness, kurtosis)
    norm = _normalization(mu, sigma, skewness, kurtosis, z_max)
    with np.errstate(all="ignore"):
        result = raw / norm
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def dual_pearson_profile(z: np.ndarray | float,
                          mu_h: float, sigma_h: float,
                          skewness_h: float, kurtosis_h: float,
                          mu_t: float, sigma_t: float,
                          skewness_t: float, kurtosis_t: float,
                          head_fraction: float,
                          z_max: float | None = None) -> np.ndarray:
    """
    Dual Pearson IV depth profile (head + tail, each normalized independently).

    head_fraction : fraction of dose in the head component (0 < f < 1)
    """
    head = pearson_profile(z, mu_h, sigma_h, skewness_h, kurtosis_h, z_max)
    tail = pearson_profile(z, mu_t, sigma_t, skewness_t, kurtosis_t, z_max)
    hf   = float(np.clip(head_fraction, 0.0, 1.0))
    return hf * head + (1.0 - hf) * tail
