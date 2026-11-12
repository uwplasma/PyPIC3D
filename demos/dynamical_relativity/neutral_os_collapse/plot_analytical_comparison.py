"""Compare the final Cartesian OS ray with stationary Schwarzschild geometry.

Adapted from RadiShPICR/demos/os_collapse_Z4C_GR/plot_analytical_comparison.py.
Native snapshots remain unchanged. A radial height function is interpreted
only under measured stationarity and spherical-symmetry assumptions.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    from diagnostics import reconstruct_metric
else:
    from .diagnostics import reconstruct_metric


DEFAULT_INPUT = Path(__file__).with_name("outputs")
INNER_RADIUS_LIMIT = 6.
ASSUMPTION_TOLERANCE = .01


def read_profiles(directory):
    profiles = []
    for path in sorted((Path(directory) / "profiles").glob("profile_*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            profiles.append({key: saved[key] for key in saved.files})
    if not profiles:
        raise FileNotFoundError(f"No Cartesian OS ray profiles in {directory}")
    return profiles


def physical_profiles(profile, prefix=""):
    return dict(alpha=profile[prefix + "alpha"], beta=profile[prefix + "beta"],
                gamma_rr=profile[prefix + "conformal_grr"]/profile[prefix + "chi"],
                gamma_T=profile[prefix + "conformal_gt"]/profile[prefix + "chi"])


def measure_assumptions(profiles, mass, R):
    """Measure ray disagreement and the largest late change at fixed code radius.

    Normalize by each field's largest absolute value in 2.5 <= R/M <= 6.
    A small shift gets a 1e-3 floor so an exactly zero shift is well defined.
    Exclude the last two stencil cells from scientific comparison summaries.
    """
    final = profiles[-1]
    region = (R/mass >= 2.5) & (R/mass <= INNER_RADIUS_LIMIT)
    region[-2:] = False
    if not np.any(region):
        return dict(supported=False, reasons=["No resolved exterior comparison interval"],
                    symmetry=None, stationarity=None)
    main = physical_profiles(final)
    rays = physical_profiles(final, "ray_")
    symmetry = {}
    for name in main:
        scale = max(float(np.max(np.abs(main[name][region]))), 1.e-3)
        symmetry[name] = float(np.max(np.abs(rays[name][:, region] - main[name][region]))/scale)
    symmetry["tangential_metric_anisotropy"] = float(np.max(final["ray_transverse_anisotropy"][:, region]))
    symmetry["radial_tangential_metric_ratio"] = float(np.max(final["ray_radial_tangent_ratio"][:, region]))
    symmetry["tangential_shift_absolute"] = float(np.max(final["ray_shift_tangent"][:, region]))

    end = float(final["time"])/mass
    start = end-5.
    late = [profile for profile in profiles if start-1.e-7 <= float(profile["time"])/mass <= end+1.e-7]
    complete = (len(late) >= 2 and float(late[0]["time"])/mass <= start+.25+1.e-7
                and float(late[-1]["time"])/mass >= end-1.e-7)
    stationarity = None
    if complete:
        stationarity = {}
        for name in main:
            scale = max(float(np.max(np.abs(main[name][region]))), 1.e-3)
            changes = [np.max(np.abs(physical_profiles(profile)[name][region] - main[name][region]))
                       for profile in late]
            stationarity[name] = float(max(changes)/scale)
    reasons = []
    if max(symmetry.values()) > ASSUMPTION_TOLERANCE:
        reasons.append("Spherical-symmetry discrepancy exceeds 1%")
    if not complete:
        reasons.append("The saved profiles do not span the final 5M interval")
    elif max(stationarity.values()) > ASSUMPTION_TOLERANCE:
        reasons.append("Late metric/gauge change exceeds 1%")
    return dict(supported=not reasons, reasons=reasons, symmetry=symmetry,
                stationarity=stationarity, threshold=ASSUMPTION_TOLERANCE,
                radius_window_over_mass=[2.5, INNER_RADIUS_LIMIT],
                time_window_over_mass=[start, end],
                normalization="Maximum absolute field in the radial window; beta floor 1e-3",
                ray_count=int(final["ray_directions"].shape[0]))


def comparison_report(profiles, mass):
    result = {key: np.asarray(value) for key, value in reconstruct_metric(profiles[-1]).items()}
    R = result["R"]
    analytical = {"g_RR": np.full_like(R, np.nan), "g_tt": np.full_like(R, np.nan)}
    exterior = (R > 2*mass) & (np.arange(R.size) >= np.argmin(R))
    analytical["g_RR"][exterior] = 1./(1.-2*mass/R[exterior])
    analytical["g_tt"][exterior] = -(1.-2*mass/R[exterior])
    assumptions = measure_assumptions(profiles, mass, R)
    report = dict(time_over_mass=float(profiles[-1]["time"])/mass,
                  initial_gravitational_mass=mass, assumptions=assumptions,
                  interpretation=("Stationary spherical overlay supported by these diagnostics"
                                  if assumptions["supported"] else "Provisional stationary spherical overlay"),
                  accuracy_claim="No convergence or boundary-independence claim from this single coarse run",
                  errors={})
    for name in analytical:
        numerical = result[name]
        valid = np.isfinite(numerical) & np.isfinite(analytical[name])
        valid[-2:] = False
        window = valid & (R/mass >= 2.5) & (R/mass <= INNER_RADIUS_LIMIT)
        relative = (numerical[window] - analytical[name][window])/analytical[name][window]
        metrics = dict(relative_rms=float(np.sqrt(np.mean(relative**2))) if relative.size else None,
                       relative_max=float(np.max(np.abs(relative))) if relative.size else None,
                       radius_window_over_mass=[2.5, INNER_RADIUS_LIMIT], probes={})
        for target in (3., 5., 10.):
            value = None
            brackets = np.flatnonzero(valid[:-1] & valid[1:]
                & (R[:-1] <= target*mass) & (R[1:] >= target*mass) & (R[1:] > R[:-1]))
            if brackets.size:
                # R need not be monotone over the full trumpet ray. Use the
                # outermost adjacent outward pair, without bridging a gap.
                left = int(brackets[-1])
                value = float(np.interp(target*mass, R[left:left+2], numerical[left:left+2]))
            exact = 1./(1.-2/target) if name == "g_RR" else -(1.-2/target)
            metrics["probes"][str(target)] = dict(numerical=value, analytical=exact,
                relative_error=(value-exact)/exact if value is not None else None,
                available=value is not None)
        report["errors"][name] = metrics
    return result, analytical, report


def save_comparison(profiles, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mass = metadata["total_mass"]
    result, analytical, report = comparison_report(profiles, mass)
    radius = result["R"]/mass
    for name, short, scale, ylabel in (("g_RR", "grr", "log", r"$g_{RR}$"),
                                        ("g_tt", "gtt", "linear", r"$g_{tt}$")):
        for extent, upper in (("inner", INNER_RADIUS_LIMIT), ("full", radius[-1])):
            figure, axis = plt.subplots(figsize=(8., 5.4))
            figure.subplots_adjust(left=.13, right=.97, top=.86, bottom=.24)
            visible = radius <= upper
            axis.plot(radius[visible], result[name][visible], label="Cartesian +x ray", linewidth=1.8)
            axis.plot(radius[visible], analytical[name][visible], "--", label="Schwarzschild, initial M", linewidth=1.8)
            axis.set(xlabel=r"Areal radius $R/M$", ylabel=ylabel, yscale=scale,
                     xlim=(2., max(2.1, min(upper, radius[-1]))),
                     title=rf"Neutral OS final metric: {extent} ($t/M={report['time_over_mass']:g}$)")
            axis.grid(alpha=.25, which="both")
            axis.legend(fontsize=9)
            figure.text(.5, .075, report["interpretation"] + ".\nOne coarse run; boundary independence and convergence unestablished.",
                        ha="center", fontsize=8)
            figure.savefig(directory / f"final_{short}_{extent}.png", dpi=180)
            plt.close(figure)
    (directory / "comparison_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    metadata = json.loads((args.input / "metadata.json").read_text())
    profiles = read_profiles(args.input)
    report = save_comparison(profiles, metadata, args.output_directory or args.input / "analytical_comparison")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
