"""
Fit Pearson IV implant parameters to a SIMS depth profile.

This example uses digitized 31P SIMS data for a 5 keV P-in-Si implant
(sims_5keV_P_Si.csv) and fits both a single and a dual Pearson IV model.
The resulting parameters are printed in ViennaPS config.txt format so they
can be pasted directly into a simulation config file.

Usage
-----
python fitSIMS.py
python fitSIMS.py --csv my_sims_data.csv --dual
"""

import argparse
import sys
from pathlib import Path

# The implant subpackage has no dependency on viennaps / the full ViennaFit
# stack, so we import it directly from the src tree.
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "viennafit"))
from implant import PearsonIVFitter, SimsProfile

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fit Pearson IV to SIMS data")
    p.add_argument("--csv",  default="sims_5keV_P_Si.csv",
                   help="Path to SIMS CSV file (default: sims_5keV_P_Si.csv)")
    p.add_argument("--dual", action="store_true",
                   help="Fit a dual Pearson IV (head + tail components)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the matplotlib plot")
    p.add_argument("--optimizer", default="scipy",
                   choices=["scipy", "dlib", "nevergrad", "cma"],
                   help="Global optimizer backend (default: scipy). "
                        "dlib/nevergrad/cma require ViennaFit's optional deps.")
    p.add_argument("--n-budget", type=int, default=1200,
                   help="Evaluation budget for the global optimizer (default 1200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


# ── Optional: customise parameter search bounds ───────────────────────────────

# Bounds for 5 keV P in Si (narrow range → faster convergence)
# Keys match the ViennaPS config.txt parameter names.
# 'skewness' bounds cover the β₂ (kurtosis) parameter range;
# 'kurtosis' bounds cover the γ₁ (skewness) parameter range.
HEAD_BOUNDS = {
    "projectedRange": (1.0, 15.0),   # Rp for 5 keV P in Si is ~4-8 nm
    "depthSigma":     (0.5,  8.0),
    "skewness":       (0.5, 30.0),   # → C++ params.beta (β₂ position in formula)
    "kurtosis":       (-2.0, 4.0),   # → C++ params.gamma (γ₁ position in formula)
    "log_amplitude":  (-3.0, 1.0),
}

TAIL_BOUNDS = {
    "tailProjectedRange": (10.0, 100.0),
    "tailDepthSigma":     ( 5.0,  60.0),
    "tailSkewness":       ( 0.3,  10.0),
    "tailKurtosis":       (-2.0,   4.0),
    "headFraction":       ( 0.5,  0.999),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load SIMS data ────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path

    print(f"Loading SIMS data from: {csv_path}")
    profile = SimsProfile.from_csv(csv_path, label="⁵ keV P in Si")
    print(f"  {len(profile)} data points, depth range "
          f"{profile.depth.min():.1f} – {profile.depth.max():.1f} nm\n")

    # ── Single Pearson IV fit ─────────────────────────────────────────────────
    print("Fitting single Pearson IV ...")
    single_fitter = PearsonIVFitter(profile, dual=False)
    single_result = single_fitter.fit(
        head_bounds=HEAD_BOUNDS,
        n_budget=args.n_budget,
        seed=args.seed,
        optimizer=args.optimizer,
    )
    single_result.print_summary()

    # ── Dual Pearson IV fit ───────────────────────────────────────────────────
    if args.dual:
        print(f"Fitting dual Pearson IV (head + tail) [{args.optimizer}] ...")
        dual_fitter = PearsonIVFitter(profile, dual=True)
        dual_result = dual_fitter.fit(
            head_bounds=HEAD_BOUNDS,
            tail_bounds=TAIL_BOUNDS,
            n_budget=args.n_budget,
            seed=args.seed,
            optimizer=args.optimizer,
        )
        dual_result.print_summary()
        best_result = dual_result
    else:
        best_result = single_result

    # ── Config output ─────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Copy these lines into your ViennaPS config.txt:")
    print("=" * 52)
    print(best_result.to_config_string())
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        if args.dual:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            single_result.plot(profile, ax=axes[0], show=False)
            axes[0].set_title("Single Pearson IV")
            dual_result.plot(profile, ax=axes[1], show=False)
            axes[1].set_title("Dual Pearson IV (head + tail)")
            plt.tight_layout()
            plt.show()
        else:
            best_result.plot(profile)


if __name__ == "__main__":
    main()
