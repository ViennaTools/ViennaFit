"""
Full simulation-calibration workflow for ion implantation Pearson IV parameters.

Two complementary steps are demonstrated:

  Step 1  (fast, no ViennaPS needed)
  ────────────────────────────────────
  PearsonIVFitter fits the Pearson IV analytical formula directly to SIMS data.
  This gives a good initial estimate in ~2 s.

  Step 2  (simulation-in-loop, requires viennaps Python bindings)
  ───────────────────────────────────────────────────────────────
  SimulationCalibrator refines the parameters by running the full ViennaPS
  implant simulation on every optimizer call and comparing the resulting cell-set
  depth profile against the SIMS target.

  Why use Step 2?
  ──────────────────────────────────────
  The analytical fitter sees only the 1-D profile shape.  The simulation
  accounts for the beam geometry, mask shadowing, screen-oxide energy loss, and
  lateral straggle of the specific device structure — effects that shift where
  and how much dopant is deposited.  Calibrating against the simulation output
  therefore gives Pearson IV moments that are self-consistent with the geometry
  the moments will be used in.

Usage
─────
  python calibrateSimulation.py                    # step 1 only (no viennaps)
  python calibrateSimulation.py --sim              # both steps (needs viennaps)
  python calibrateSimulation.py --sim --optimizer dlib
  python calibrateSimulation.py --sim --n-budget 150 --optimizer nevergrad
"""

import argparse
import functools
import sys
from pathlib import Path

# ── Path setup — run without installing the package ───────────────────────────
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "viennafit"))
from implant import (
    FitResult,
    PearsonIVFitter,
    SimsProfile,
    SimulationCalibrator,
    build_blanket_substrate_domain,
    # build_masked_substrate_domain,  # use this instead for patterned-device calibration
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-step Pearson IV calibration: analytical + simulation")
    p.add_argument("--csv", default="sims_5keV_P_Si.csv",
                   help="SIMS CSV file (default: sims_5keV_P_Si.csv)")
    p.add_argument("--sim", action="store_true",
                   help="Run step 2: simulation-in-loop refinement "
                        "(requires viennaps Python bindings)")
    p.add_argument("--optimizer", default="scipy",
                   choices=["scipy", "dlib", "nevergrad", "cma"],
                   help="Optimizer backend for both steps (default: scipy)")
    p.add_argument("--n-budget", type=int, default=200,
                   help="Simulation calls for step 2 (default: 200)")
    p.add_argument("--n-analytical", type=int, default=1200,
                   help="Evaluations for step 1 analytical fit (default: 1200)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Domain geometry ─────────────────────────────────────────────────────────
# The SIMS data (sims_5keV_P_Si.csv) is from a bare-wafer (blanket) implant,
# so the calibration domain must also be unmasked.
# Use build_blanket_substrate_domain: Si slab + optional screen oxide, no mask.
#
# Use build_masked_substrate_domain only when calibrating against measurements
# from a patterned device (e.g. TEM junction depth vs mask opening width).
BLANKET_DOMAIN_KWARGS = dict(
    grid_delta      = 1.0,    # nm — coarser grid speeds up calibration
    x_extent        = 50.0,   # nm — narrower is fine, no lateral variation
    top_space       = 10.0,   # nm
    substrate_depth = 80.0,   # nm  (5 keV range ~ 5-7 nm, tail ~40 nm)
    oxide_thickness =  2.0,   # nm  screen oxide
)

# Bounds tailored for 5 keV P in Si
HEAD_BOUNDS = {
    "projectedRange": (1.0, 15.0),
    "depthSigma":     (0.5,  8.0),
    "skewness":       (0.5, 30.0),
    "kurtosis":       (-2.0, 4.0),
    "log_amplitude":  (-3.0, 1.0),
}
TAIL_BOUNDS = {
    "tailProjectedRange": (10.0, 80.0),
    "tailDepthSigma":     (5.0, 50.0),
    "tailSkewness":       (0.5, 15.0),
    "tailKurtosis":       (-2.0, 4.0),
    "headFraction":       (0.5, 0.999),
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    csv_path = Path(__file__).parent / args.csv
    print(f"Loading SIMS data: {csv_path}")
    profile = SimsProfile.from_csv(csv_path, label="31P 5 keV")
    print(f"  {len(profile)} points, "
          f"{profile.depth.min():.1f} – {profile.depth.max():.1f} nm\n")

    # ─── Step 1: fast analytical fit ──────────────────────────────────────────
    print("━" * 56)
    print("  Step 1 — Analytical Pearson IV fit (no simulation)")
    print("━" * 56)
    fitter = PearsonIVFitter(profile, dual=True)
    fast = fitter.fit(
        head_bounds=HEAD_BOUNDS,
        tail_bounds=TAIL_BOUNDS,
        n_budget=args.n_analytical,
        optimizer=args.optimizer,
        seed=args.seed,
    )
    fast.print_summary()

    if not args.sim:
        print("(Pass --sim to also run the simulation-in-loop step 2)")
        if not args.no_plot:
            fast.plot(profile)
        return

    # ─── Step 2: simulation-in-loop refinement ────────────────────────────────
    print("━" * 56)
    print("  Step 2 — Simulation-in-loop refinement")
    print(f"           optimizer={args.optimizer}  budget={args.n_budget}")
    print("━" * 56)

    # Blanket (unmasked) domain — matches bare-wafer SIMS calibration data.
    # Switch to build_masked_substrate_domain + MASKED_DOMAIN_KWARGS when
    # calibrating against measurements from a patterned device.
    domain_factory = functools.partial(
        build_blanket_substrate_domain, **BLANKET_DOMAIN_KWARGS)

    calibrator = SimulationCalibrator(
        sims_profile     = profile,
        domain_factory   = domain_factory,
        species          = "P",
        dose_cm2         = 1e14,
        tilt_angle       = 6.,
        screen_thickness = 2.,      # nm — screen oxide subtracts from Rp
        dual             = True,
        log_fit          = True,
    )

    sim = calibrator.fit(
        initial_guess = fast,        # warm-start DE from step-1 result
        head_bounds   = HEAD_BOUNDS,
        tail_bounds   = TAIL_BOUNDS,
        n_budget      = args.n_budget,
        optimizer     = args.optimizer,
        seed          = args.seed,
    )
    sim.print_summary()

    # ─── Compare ──────────────────────────────────────────────────────────────
    print("━" * 56)
    print("  Comparison: analytical  vs  simulation-calibrated")
    print("━" * 56)
    fmt = "  {:30s}  {:>10.4f}  {:>10.4f}"
    print(f"  {'Parameter':<30}  {'Analytical':>10}  {'Sim-calib':>10}")
    print("  " + "-" * 52)
    params = [
        ("projectedRange (nm)",   fast.projectedRange,   sim.projectedRange),
        ("depthSigma (nm)",       fast.depthSigma,       sim.depthSigma),
        ("skewness",              fast.skewness,         sim.skewness),
        ("kurtosis",              fast.kurtosis,         sim.kurtosis),
        ("headFraction",          fast.headFraction,     sim.headFraction),
        ("tailProjectedRange (nm)", fast.tailProjectedRange, sim.tailProjectedRange),
        ("tailDepthSigma (nm)",   fast.tailDepthSigma,   sim.tailDepthSigma),
        ("R² (log-space)",        fast.r2_log,           sim.r2_log),
        ("RMSE (log)",            fast.rmse_log,         sim.rmse_log),
    ]
    for name, a, b in params:
        if a is not None and b is not None:
            print(fmt.format(name, a, b))
    print()

    print("━" * 56)
    print("  Simulation-calibrated config.txt block:")
    print("━" * 56)
    print(sim.to_config_string())

    # ─── Plot ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fast.plot(profile, ax=axes[0], show=False)
        axes[0].set_title("Step 1: analytical Pearson IV")
        sim.plot(profile, ax=axes[1], show=False)
        axes[1].set_title("Step 2: simulation-calibrated")
        plt.suptitle("Pearson IV calibration — 5 keV P in Si", y=1.01)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
