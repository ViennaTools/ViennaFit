"""Shared helpers for SIMS fitting and ViennaPS calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

Optimizer = Literal["scipy", "dlib", "nevergrad", "cma"]
OPTIMIZERS = ("scipy", "dlib", "nevergrad", "cma")


def sims_depth_from_center(
    center: Sequence[float],
    depth_axis: int = 1,
    surface_position: float = 0.0,
) -> float:
    """Return positive SIMS depth into the substrate from a ViennaPS cell center."""
    return float(surface_position - center[depth_axis])


def residual_loss(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    floor: float,
    log_fit: bool = True,
    reduction: Literal["sum", "mean"] = "sum",
) -> float:
    """Residual loss for linear or log-space profile comparison."""
    pred = np.maximum(np.asarray(predicted, dtype=np.float64), floor)
    obs = np.maximum(np.asarray(observed, dtype=np.float64), floor)
    if log_fit:
        diff = np.log10(pred) - np.log10(obs)
    else:
        diff = pred - obs
    value = np.mean(diff * diff) if reduction == "mean" else np.sum(diff * diff)
    return float(value)


def r2_log(predicted: np.ndarray, observed: np.ndarray, floor: float) -> float:
    """R-squared computed in log10 space."""
    obs = np.log10(np.maximum(np.asarray(observed, dtype=np.float64), floor))
    pred = np.log10(np.maximum(np.asarray(predicted, dtype=np.float64), floor))
    ss_res = float(np.sum((pred - obs) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def rmse_log(predicted: np.ndarray, observed: np.ndarray, floor: float) -> float:
    """Root-mean-square error computed in log10 space."""
    obs = np.log10(np.maximum(np.asarray(observed, dtype=np.float64), floor))
    pred = np.log10(np.maximum(np.asarray(predicted, dtype=np.float64), floor))
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


@dataclass
class _BudgetedObjective:
    objective: Callable[[np.ndarray], float]
    budget: int
    penalty: float = 1.0e30

    calls: int = 0
    best_value: float = float("inf")
    best_x: Optional[np.ndarray] = None

    def __call__(self, x) -> float:
        arr = np.asarray(x, dtype=np.float64)
        if self.calls >= self.budget:
            return self.penalty

        self.calls += 1
        value = float(self.objective(arr))
        if not np.isfinite(value):
            value = self.penalty

        if value < self.best_value:
            self.best_value = value
            self.best_x = arr.copy()
        return value


def _clip_to_bounds(x, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(arr, lo, hi)


def _initial_population(
    bounds: Sequence[tuple[float, float]],
    seed: int,
    initial_guess: Optional[Sequence[float]],
    n_budget: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    size = max(5, min(max(n_budget, 5), max(5, 2 * dim + 1)))
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    pop = rng.uniform(lo, hi, size=(size, dim))
    if initial_guess is not None:
        pop[0] = _clip_to_bounds(initial_guess, bounds)
    return pop


def run_optimizer(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    n_budget: int,
    seed: int,
    optimizer: Optimizer,
    *,
    initial_guess: Optional[Sequence[float]] = None,
    polish: bool = True,
) -> np.ndarray:
    """
    Run an optimizer with ``n_budget`` interpreted as objective-call budget.

    The scipy backend uses a budget-capped objective wrapper, so expensive
    simulation objectives are not called far beyond the requested budget even
    though scipy itself may request additional penalty-valued evaluations.
    """
    if optimizer not in _BACKENDS:
        raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}; got '{optimizer}'")
    if n_budget < 1:
        raise ValueError("n_budget must be at least 1")
    if not bounds:
        raise ValueError("bounds must not be empty")

    return _BACKENDS[optimizer](
        objective,
        list(bounds),
        int(n_budget),
        int(seed),
        initial_guess=initial_guess,
        polish=polish,
    )


def _run_scipy(
    objective,
    bounds: list,
    n_budget: int,
    seed: int,
    *,
    initial_guess=None,
    polish: bool = True,
) -> np.ndarray:
    wrapped = _BudgetedObjective(objective, n_budget)
    dim = len(bounds)
    popsize = max(1, min(5, max(n_budget // max(dim, 1), 1)))
    initial_eval = max(popsize * dim, 1)
    maxiter = max(0, (n_budget // initial_eval) - 1)

    init = _initial_population(bounds, seed, initial_guess, n_budget)
    result = differential_evolution(
        wrapped,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        seed=seed,
        polish=False,
        workers=1,
        init=init,
    )

    best_x = wrapped.best_x if wrapped.best_x is not None else result.x
    if polish and wrapped.calls < n_budget:
        remaining = max(1, n_budget - wrapped.calls)
        local = minimize(
            wrapped,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxfun": remaining,
                "maxiter": remaining,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )
        if wrapped.best_x is None and local.x is not None:
            best_x = local.x
        else:
            best_x = wrapped.best_x
    return _clip_to_bounds(best_x, bounds)


def _run_dlib(
    objective,
    bounds: list,
    n_budget: int,
    seed: int,
    *,
    initial_guess=None,
    polish: bool = True,
) -> np.ndarray:
    try:
        import dlib
    except ImportError:
        raise ImportError(
            "dlib is required for optimizer='dlib'. Install it with: pip install dlib-bin"
        )

    wrapped = _BudgetedObjective(objective, n_budget)
    if initial_guess is not None:
        wrapped(_clip_to_bounds(initial_guess, bounds))

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    remaining = max(1, n_budget - wrapped.calls)
    result, _ = dlib.find_min_global(
        lambda *args: wrapped(np.array(args, dtype=np.float64)),
        lo,
        hi,
        remaining,
    )
    best_x = wrapped.best_x if wrapped.best_x is not None else np.array(result)
    return _clip_to_bounds(best_x, bounds)


def _run_nevergrad(
    objective,
    bounds: list,
    n_budget: int,
    seed: int,
    *,
    initial_guess=None,
    polish: bool = True,
) -> np.ndarray:
    try:
        import nevergrad as ng
    except ImportError:
        raise ImportError(
            "nevergrad is required for optimizer='nevergrad'. Install it with: pip install nevergrad"
        )

    wrapped = _BudgetedObjective(objective, n_budget)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    param = ng.p.Array(shape=(len(bounds),)).set_bounds(lower=lo, upper=hi)
    opt = ng.optimizers.DE(parametrization=param, budget=n_budget)
    opt.parametrization.random_state = np.random.RandomState(seed)

    if initial_guess is not None:
        guess = _clip_to_bounds(initial_guess, bounds)
        wrapped(guess)
        opt.tell(param.spawn_child(new_value=guess), wrapped.best_value)

    def ng_objective(x: ng.p.Array) -> float:
        return wrapped(x.value)

    recommendation = opt.minimize(ng_objective)
    best_x = wrapped.best_x if wrapped.best_x is not None else recommendation.value
    if polish and wrapped.calls < n_budget:
        remaining = max(1, n_budget - wrapped.calls)
        minimize(
            wrapped,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": remaining, "maxiter": remaining},
        )
        best_x = wrapped.best_x
    return _clip_to_bounds(best_x, bounds)


def _run_cma(
    objective,
    bounds: list,
    n_budget: int,
    seed: int,
    *,
    initial_guess=None,
    polish: bool = True,
) -> np.ndarray:
    try:
        import cma
    except ImportError:
        raise ImportError(
            "cma is required for optimizer='cma'. Install it with: pip install cma"
        )

    wrapped = _BudgetedObjective(objective, n_budget)
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    x0 = (_clip_to_bounds(initial_guess, bounds).tolist()
          if initial_guess is not None
          else [(l + h) / 2.0 for l, h in bounds])
    sigma0 = 0.25 * max(h - l for l, h in bounds)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {"bounds": [lo, hi], "maxfevals": n_budget, "seed": seed, "verbose": -9},
    )
    es.optimize(wrapped)
    best_x = wrapped.best_x if wrapped.best_x is not None else np.array(es.result.xbest)
    if polish and wrapped.calls < n_budget:
        remaining = max(1, n_budget - wrapped.calls)
        minimize(
            wrapped,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": remaining, "maxiter": remaining},
        )
        best_x = wrapped.best_x
    return _clip_to_bounds(best_x, bounds)


_BACKENDS = {
    "scipy": _run_scipy,
    "dlib": _run_dlib,
    "nevergrad": _run_nevergrad,
    "cma": _run_cma,
}
