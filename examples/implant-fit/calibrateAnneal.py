"""
Three-input anneal calibration workflow.

Given three measurements from a blanket-wafer implant experiment:

  1. Post-implant SIMS        — shape of the as-implanted dopant profile
  2. Post-anneal SIMS         — shape of the diffused dopant profile
  3. Sheet resistance (Rsh)   — electrically active dose integral

This script calibrates the ViennaPS Anneal model parameters so that a
simulation started from the measured pre-anneal profile reproduces both
the post-anneal SIMS shape and the measured Rsh.

Calibrated parameters
---------------------
  annealD0, annealEa       — Arrhenius dopant diffusivity (profile shape)
  annealSolidSolubilityC0,
  annealSolidSolubilityEa  — Arrhenius solid solubility (Rsh, active fraction)
                             (only fitted when --rsh is provided)

Workflow
--------
  python calibrateAnneal.py --rsh 1200
      Full calibration: profile shape + solid solubility (requires viennaps)

  python calibrateAnneal.py --rsh 1200 --optimizer dlib --n-budget 150
      Same with dlib optimizer

  python calibrateAnneal.py --no-sim
      Dry-run: print expected config.txt keys without running the simulation
      (useful for checking the workflow structure)

The viennaps Python bindings are required for all calibration modes.
Build ViennaPS with -DVIENNAPS_BUILD_PYTHON=ON.
"""

import argparse
import functools
import sys
from pathlib import Path

# ── Path setup — run without installing the package ───────────────────────────
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "viennafit"))
from implant import (
    AnnealCalibrator,
    AnnealFitResult,
    SimsProfile,
    build_blanket_substrate_domain,
    # Note: sheet resistance is computed by vps.SheetResistance (ViennaCS),
    # not by ViennaFit.  AnnealCalibrator calls it internally when rsh_target
    # is set; you can also call it directly on any domain after simulation.
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Anneal parameter calibration: SIMS profile + sheet resistance")
    p.add_argument("--pre-csv",   default="sims_5keV_P_Si.csv",
                   help="Pre-anneal (post-implant) SIMS CSV  (default: %(default)s)")
    p.add_argument("--post-csv",  default="sims_5keV_P_Si_postanneal.csv",
                   help="Post-anneal SIMS CSV  (default: %(default)s)")
    p.add_argument("--rsh",       type=float, default=1200.0,
                   help="Measured sheet resistance [Ω/□]  (default: %(default)s). "
                        "Pass 0 to calibrate profile shape only.")
    p.add_argument("--rsh-weight", type=float, default=0.4,
                   help="Weight of Rsh term in combined objective [0,1]  "
                        "(default: %(default)s)")
    p.add_argument("--anneal-temp", type=float, default=1000.0,
                   help="Anneal temperature [°C]  (default: %(default)s)")
    p.add_argument("--anneal-time", type=float, default=30.0,
                   help="Anneal duration [s]  (default: %(default)s)")
    p.add_argument("--dose",      type=float, default=1e14,
                   help="Implant dose [cm⁻²]  (default: %(default)s)")
    p.add_argument("--optimizer", default="scipy",
                   choices=["scipy", "dlib", "nevergrad", "cma"],
                   help="Optimizer backend  (default: %(default)s)")
    p.add_argument("--n-budget",  type=int, default=200,
                   help="Simulation budget  (default: %(default)s)")
    p.add_argument("--no-sim",    action="store_true",
                   help="Skip simulation — dry-run only")
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ── Domain geometry ───────────────────────────────────────────────────────────
# Blanket (unmasked) substrate — matches bare-wafer SIMS measurement conditions.
# 5 keV P in Si: range ~5–7 nm, post-anneal tail ~60–80 nm → 80 nm deep domain.
DOMAIN_KWARGS = dict(
    grid_delta      = 1.0,    # nm
    x_extent        = 50.0,   # nm  (no lateral variation needed for 1-D Rsh)
    top_space       = 10.0,   # nm
    substrate_depth = 80.0,   # nm
    oxide_thickness =  2.0,   # nm  screen oxide
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    here = Path(__file__).parent

    # ── Load SIMS data ────────────────────────────────────────────────────────
    pre_path  = here / args.pre_csv
    post_path = here / args.post_csv

    print(f"Loading pre-anneal SIMS:  {pre_path.name}")
    pre_anneal = SimsProfile.from_csv(pre_path, label="Pre-anneal (post-implant)")
    print(f"  {len(pre_anneal)} points, "
          f"{pre_anneal.depth.min():.1f}–{pre_anneal.depth.max():.1f} nm")

    print(f"Loading post-anneal SIMS: {post_path.name}")
    post_anneal = SimsProfile.from_csv(post_path, label="Post-anneal (target)")
    print(f"  {len(post_anneal)} points, "
          f"{post_anneal.depth.min():.1f}–{post_anneal.depth.max():.1f} nm")

    rsh_target = float(args.rsh) if args.rsh > 0 else None
    if rsh_target is not None:
        print(f"Target Rsh: {rsh_target:.1f} Ω/□")
    print()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.no_sim:
        _print_config_keys(rsh_target)
        return

    # ── Calibration ───────────────────────────────────────────────────────────
    print("━" * 60)
    print("  Anneal calibration (ViennaPS simulation in loop)")
    print(f"  Anneal: {args.anneal_temp:.0f} °C / {args.anneal_time:.0f} s")
    print(f"  optimizer={args.optimizer}  budget={args.n_budget}")
    if rsh_target is not None:
        print(f"  Fitting: D0, Ea  +  solid solubility (Rsh target)")
    else:
        print(f"  Fitting: D0, Ea  (profile shape only, no Rsh target)")
    print("━" * 60)

    domain_factory = functools.partial(
        build_blanket_substrate_domain, **DOMAIN_KWARGS)

    calibrator = AnnealCalibrator(
        pre_anneal_sims  = pre_anneal,
        post_anneal_sims = post_anneal,
        rsh_target       = rsh_target,
        anneal_temp_C    = args.anneal_temp,
        anneal_time_s    = args.anneal_time,
        domain_factory   = domain_factory,
        species          = "P",
        dose_cm2         = args.dose,
        rsh_weight       = args.rsh_weight,
        log_fit          = True,
    )

    result = calibrator.fit(
        n_budget  = args.n_budget,
        optimizer = args.optimizer,
        seed      = args.seed,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print("━" * 60)
    print("  Calibrated anneal parameters:")
    print("━" * 60)
    result.print_summary()
    print()

    print("━" * 60)
    print("  config.txt block:")
    print("━" * 60)
    print(result.to_config_string())
    print()

    # When no rsh_target was provided, Rsh is not in the result.
    # To compute it post-hoc, use vps.SheetResistance on the domain directly:
    #   sr = vps.SheetResistance()
    #   sr.setCellSet(domain.getCellSet())
    #   sr.setConcentrationLabel("P_active")
    #   print(sr.computeElectron())
    # (requires keeping the domain object alive after calibration)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        result.plot(pre_anneal, post_anneal)


def _print_config_keys(rsh_target):
    """Print the config.txt keys that will be calibrated (dry-run mode)."""
    print("━" * 60)
    print("  Parameters calibrated by AnnealCalibrator")
    print("━" * 60)
    print("  # Dopant diffusivity — fitted to post-anneal SIMS profile shape")
    print("  annealD0=<pre-exponential [nm²/s]>")
    print("  annealEa=<activation energy [eV]>")
    if rsh_target is not None:
        print()
        print("  # Solid solubility — fitted to sheet resistance")
        print("  annealSolidActivation=1")
        print("  annealSolidSolubilityC0=<pre-exponential [nm⁻³]>")
        print("  annealSolidSolubilityEa=<activation energy [eV]>")
    print()
    print("  Fixed (supply from implant calibration or modeldb):")
    print("  annealInterstitialDiffusivity=...")
    print("  annealVacancyDiffusivity=...")
    print("  annealInterstitialEqC0=..., annealInterstitialEqEa=...")
    print("  annealVacancyEqC0=..., annealVacancyEqEa=...")
    print("  annealClusterKfc=..., annealClusterKr=..., annealClusterInitFraction=...")


if __name__ == "__main__":
    main()
