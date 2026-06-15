"""
viennafit.implant — Pearson IV fitting and simulation calibration for SIMS profiles.

Two complementary workflows
----------------------------

**Fast analytical fit** (pure Python, no ViennaPS needed):

    from viennafit.implant import SimsProfile, PearsonIVFitter

    profile = SimsProfile.from_csv("sims_5keV_P_Si.csv")
    result  = PearsonIVFitter(profile, dual=True).fit()
    result.print_summary()
    result.plot(profile)
    print(result.to_config_string())

**Simulation-in-the-loop implant calibration** (requires viennaps Python bindings):

    from viennafit.implant import (SimsProfile, SimulationCalibrator,
                                   build_masked_substrate_domain)
    import functools

    domain_factory = functools.partial(
        build_masked_substrate_domain,
        grid_delta=1., x_extent=200., top_space=15.,
        substrate_depth=80., opening_width=100.,
        mask_height=20., oxide_thickness=2.,
    )
    cal    = SimulationCalibrator(profile, domain_factory, dose_cm2=1e14)
    result = cal.fit(initial_guess=fast_result, n_budget=200)
    print(result.to_config_string())

**Anneal calibration from post-anneal SIMS + sheet resistance**
(requires viennaps Python bindings):

    from viennafit.implant import (SimsProfile, AnnealCalibrator,
                                   build_blanket_substrate_domain)
    import functools

    pre   = SimsProfile.from_csv("sims_pre.csv")
    post  = SimsProfile.from_csv("sims_post.csv")
    domain_factory = functools.partial(
        build_blanket_substrate_domain,
        grid_delta=1., x_extent=50., top_space=10.,
        substrate_depth=80., oxide_thickness=2.,
    )
    cal = AnnealCalibrator(
        pre_anneal_sims=pre, post_anneal_sims=post,
        rsh_target=1200.,    # Ω/□  — None to skip Rsh term
        anneal_temp_C=1000., anneal_time_s=30.,
        domain_factory=domain_factory, dose_cm2=1e14,
    )
    result = cal.fit(n_budget=200)
    result.print_summary()
    print(result.to_config_string())
"""

from .simsfit import FitResult, PearsonIVFitter, SimsProfile
from .pearsoniv import (
    dual_pearson_profile,
    pearson_profile,
    pearson_unnorm,
)
from .simprofile import (
    build_blanket_substrate_domain,
    build_masked_substrate_domain,
    extract_depth_profile,
)
from .simcalibrator import SimulationCalibrator
from .annealfit import (
    AnnealCalibrator,
    AnnealFitResult,
    init_domain_from_sims,
)

__all__ = [
    # Data + analytical fitting
    "SimsProfile",
    "PearsonIVFitter",
    "FitResult",
    # Pearson IV math
    "pearson_profile",
    "dual_pearson_profile",
    "pearson_unnorm",
    # ViennaPS utilities (require viennaps bindings)
    "extract_depth_profile",
    "build_blanket_substrate_domain",   # bare wafer — for SIMS calibration
    "build_masked_substrate_domain",    # masked device geometry
    "SimulationCalibrator",
    # Anneal calibration (Rsh via vps.SheetResistance / vcs.SheetResistance)
    "AnnealCalibrator",
    "AnnealFitResult",
    "init_domain_from_sims",
]
