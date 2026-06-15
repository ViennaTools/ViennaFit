from pathlib import Path
from functools import partial

import numpy as np
import pytest

from viennafit.implant import (
    AnnealCalibrator,
    FitResult,
    PearsonIVFitter,
    SimulationCalibrator,
    SimsProfile,
    build_blanket_substrate_domain,
    extract_depth_profile,
    pearson_profile,
)
from viennafit.implant._common import (
    residual_loss,
    run_optimizer,
    sims_depth_from_center,
)
from viennafit.implant.annealfit import init_domain_from_sims


DATA_DIR = Path(__file__).parents[1] / "examples" / "implant-fit"


class FakeCellSet:
    def __init__(self, centers, data, delta=1.0):
        self.centers = centers
        self.data = dict(data)
        self.delta = delta

    def getNumberOfCells(self):
        return len(self.centers)

    def getGridDelta(self):
        return self.delta

    def getScalarData(self, label):
        return self.data.get(label)

    def getCellCenter(self, idx):
        return self.centers[idx]

    def addScalarData(self, label, value):
        self.data[label] = [value] * len(self.centers)

    def setScalarData(self, label, values):
        self.data[label] = list(values)


class FakeDomain:
    def __init__(self, cell_set):
        self.cell_set = cell_set

    def getCellSet(self):
        return self.cell_set


def test_sims_profile_load_and_normalize():
    profile = SimsProfile.from_csv(DATA_DIR / "sims_5keV_P_Si.csv")

    assert len(profile) == 28
    assert np.all(profile.depth >= 0.0)
    assert np.all(profile.intensity > 0.0)
    assert profile.normalize().intensity.max() == pytest.approx(1.0)


def test_pearson_profile_is_finite_and_nonnegative():
    z = np.linspace(0.0, 50.0, 101)
    profile = pearson_profile(z, 6.0, 3.0, 20.0, 2.0)

    assert np.all(np.isfinite(profile))
    assert np.all(profile >= 0.0)


def test_sims_depth_transform_and_substrate_filtering():
    assert sims_depth_from_center([0.0, -10.0]) == pytest.approx(10.0)
    assert sims_depth_from_center([0.0, 2.0]) == pytest.approx(-2.0)

    centers = [[0.0, -10.0], [1.0, -10.0], [0.0, 2.0]]
    domain = FakeDomain(FakeCellSet(centers, {"P_total": [1.0, 3.0, 99.0]}))
    depths, values = extract_depth_profile(domain, "P_total")

    assert depths.tolist() == [10.0]
    assert values.tolist() == [3.0]


def test_init_domain_from_sims_writes_into_negative_y_substrate():
    centers = [[0.0, -1.0], [0.0, -5.0], [0.0, 2.0]]
    cs = FakeCellSet(centers, {})
    domain = FakeDomain(cs)

    init_domain_from_sims(
        domain,
        "P",
        depth_nm=np.array([0.0, 10.0]),
        conc=np.array([1.0, 1.0]),
        dose_cm2=1e14,
    )

    values = cs.getScalarData("P_total")
    assert values[0] > 0.0
    assert values[1] > 0.0
    assert values[2] == 0.0


def test_analytical_single_and_dual_fit_tiny_budget():
    profile = SimsProfile.from_csv(DATA_DIR / "sims_5keV_P_Si.csv")

    single = PearsonIVFitter(profile, dual=False).fit(n_budget=5, seed=1)
    dual = PearsonIVFitter(profile, dual=True).fit(n_budget=5, seed=1)

    assert np.isfinite(single.rmse_log)
    assert np.isfinite(dual.rmse_log)
    assert single.n_eval <= 5
    assert dual.n_eval <= 5


def test_shared_metrics_and_scipy_budget():
    calls = []

    def objective(x):
        calls.append(np.asarray(x).copy())
        return float((x[0] - 0.25) ** 2)

    best = run_optimizer(
        objective,
        [(0.0, 1.0)],
        n_budget=7,
        seed=3,
        optimizer="scipy",
        initial_guess=[0.2],
    )

    assert len(calls) <= 7
    assert 0.0 <= best[0] <= 1.0
    assert residual_loss(np.array([1.0]), np.array([1.0]), floor=1e-6) == 0.0


def test_blanket_domain_profile_with_viennaps_if_available():
    pytest.importorskip("viennaps")

    domain = build_blanket_substrate_domain(
        grid_delta=2.0,
        x_extent=10.0,
        top_space=4.0,
        substrate_depth=10.0,
        oxide_thickness=0.0,
    )
    init_domain_from_sims(
        domain,
        "P",
        depth_nm=np.array([0.0, 10.0]),
        conc=np.array([1.0, 0.5]),
        dose_cm2=1e14,
    )
    depths, values = extract_depth_profile(domain, "P_total")

    assert len(depths) > 0
    assert depths.min() >= 0.0
    assert values.max() > 0.0


def test_simulation_calibrator_smoke_with_viennaps_if_available():
    pytest.importorskip("viennaps")

    profile = SimsProfile(
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        np.array([1.0, 0.8, 0.45, 0.2, 0.08]),
    )
    domain_factory = partial(
        build_blanket_substrate_domain,
        grid_delta=2.0,
        x_extent=12.0,
        top_space=4.0,
        substrate_depth=20.0,
        oxide_thickness=0.0,
    )
    initial = FitResult(
        projectedRange=4.0,
        depthSigma=2.0,
        skewness=10.0,
        kurtosis=1.0,
        amplitude=1.0,
    )
    calibrator = SimulationCalibrator(
        profile,
        domain_factory,
        dual=False,
        dose_cm2=1e12,
        tilt_angle=0.0,
    )

    result = calibrator.fit(
        initial_guess=initial,
        head_bounds={
            "projectedRange": (3.5, 4.5),
            "depthSigma": (1.5, 2.5),
            "skewness": (8.0, 12.0),
            "kurtosis": (0.5, 1.5),
            "log_amplitude": (-0.5, 0.5),
        },
        n_budget=1,
    )

    assert result.n_eval == 1
    assert result.projectedRange == pytest.approx(4.0)
    assert np.isfinite(result.rmse_log)


def test_anneal_calibrator_smoke_with_viennaps_if_available():
    pytest.importorskip("viennaps")

    pre = SimsProfile(
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        np.array([1.0, 0.7, 0.4, 0.2, 0.08]),
        label="pre",
    )
    post = SimsProfile(
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        np.array([0.9, 0.75, 0.5, 0.3, 0.12]),
        label="post",
    )
    domain_factory = partial(
        build_blanket_substrate_domain,
        grid_delta=2.0,
        x_extent=12.0,
        top_space=4.0,
        substrate_depth=20.0,
        oxide_thickness=0.0,
    )
    calibrator = AnnealCalibrator(
        pre_anneal_sims=pre,
        post_anneal_sims=post,
        rsh_target=1.0e3,
        anneal_temp_C=1000.0,
        anneal_time_s=0.1,
        domain_factory=domain_factory,
        dose_cm2=1e12,
        rsh_weight=0.1,
    )

    result = calibrator.fit(
        bounds={
            "log10_D0": (-4.0, -3.9),
            "annealEa": (0.9, 1.0),
            "log10_C0_ss": (2.0, 2.1),
            "Ea_ss": (0.0, 0.1),
        },
        n_budget=1,
    )

    assert result.n_eval == 1
    assert result.sim_profile is not None
    assert result.sim_profile.max() > 0.0
    assert result.rsh_computed is not None
