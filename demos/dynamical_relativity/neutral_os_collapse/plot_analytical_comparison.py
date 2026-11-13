"""Compare the final Cartesian OS ray with stationary Schwarzschild geometry.

Adapted from RadiShPICR/demos/os_collapse_Z4C_GR/plot_analytical_comparison.py.
Native snapshots remain unchanged. A radial height function is interpreted
only under measured stationarity and spherical-symmetry assumptions.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import openpmd_api as io
import toml
from jax.scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    from diagnostics import outward_branch, reconstruct_metric
    from initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS
else:
    from .diagnostics import outward_branch, reconstruct_metric
    from .initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS


CONFIG = Path(__file__).with_name("os_collapse.toml")
INNER_RADIUS_LIMIT = 6.
ASSUMPTION_TOLERANCE = .01
METRIC_COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")


def read_configuration(path=CONFIG):
    """Take the run directory and the prescribed mass from the demo inputs."""
    config = toml.load(path)
    parameters = {**config.get("simulation_parameters", {}),
                  **config.get("static_parameters", {})}
    return dict(output_dir=Path(parameters.get("output_dir", ".")),
                total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS)


def ray_geometry(shape, origin, spacing):
    """Six axis rays and eight body diagonals at common radii.

    This reproduces the sampling the demo used while it still owned a tiled
    in-run diagnostic: the first ray is +x and supplies the short names the
    plotters below index. Linear interpolation includes the true coordinate
    origin even when it lies between nodes.
    """
    axes = jnp.asarray(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                        (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)))
    diagonals = jnp.asarray(((1., 1., 1.), (1., 1., -1.), (1., -1., 1.),
                             (1., -1., -1.), (-1., 1., 1.), (-1., 1., -1.),
                             (-1., -1., 1.), (-1., -1., -1.))) / jnp.sqrt(3.)
    directions = jnp.concatenate((axes, diagonals))
    origin = jnp.asarray(origin)
    upper = origin + (jnp.asarray(shape) - 1) * spacing
    radial_limit = jnp.min(jnp.minimum(-origin, upper))
    r = jnp.linspace(0., radial_limit, (min(shape) - 1) // 2 + 1)
    coordinates = (directions[:, :, None] * r - origin[None, :, None]) / spacing
    # Rays stop at the last node, so clamping only guards against round-off.
    coordinates = jnp.clip(coordinates, 0., (jnp.asarray(shape) - 1.)[None, :, None])
    return directions, r, coordinates


def sample_rays(field, coordinates):
    """Linearly sample a global C-node array along every ray.

    ``field`` may carry leading component axes; they are preserved ahead of the
    (ray, radius) axes so the projections below can contract them directly.
    """
    field = jnp.asarray(field)
    leading = field.shape[:-3]
    flat = field.reshape((-1,) + field.shape[-3:])
    values = jax.vmap(lambda component: jax.vmap(lambda ray: map_coordinates(
        component, ray, order=1, mode="constant", cval=0.))(coordinates))(flat)
    return values.reshape(leading + values.shape[1:])


def build_profile(meshes, shape, origin, spacing):
    """Project sampled Cartesian geometry onto radial and tangential parts.

    The evolved conformal factor is W; saved ``chi`` is W**2. The radial
    components are projections of the Cartesian metric, not a radial solve.
    """
    directions, r, coordinates = ray_geometry(shape, origin, spacing)
    alpha = sample_rays(meshes["lapse"], coordinates)
    W = sample_rays(meshes["conformal_factor"], coordinates)
    shift = sample_rays(meshes["shift"], coordinates)
    metric = sample_rays(meshes["conformal_metric"], coordinates)

    radial_metric = jnp.einsum("ai,ijar,aj->ar", directions, metric, directions)
    tangential_metric = (jnp.einsum("iiar->ar", metric) - radial_metric) / 2.
    beta = jnp.einsum("ai,iar->ar", directions, shift)

    projector = jnp.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    tangential_tensor = jnp.einsum("aij,jkar,akl->ilar", projector, metric, projector)
    tangential_anisotropy = tangential_tensor - jnp.einsum(
        "ar,aij->ijar", tangential_metric, projector)
    radial_tangent = jnp.einsum("aij,jkar,ak->iar", projector, metric, directions)
    shift_tangent = shift - jnp.einsum("ai,ar->iar", directions, beta)
    result = dict(r=r, ray_directions=directions,
                  ray_alpha=alpha, ray_beta=beta, ray_chi=W**2,
                  ray_conformal_grr=radial_metric, ray_conformal_gt=tangential_metric,
                  ray_transverse_anisotropy=jnp.sqrt(jnp.sum(tangential_anisotropy**2, axis=(0, 1)))
                    / (jnp.sqrt(2.) * jnp.abs(tangential_metric)),
                  ray_radial_tangent_ratio=jnp.sqrt(jnp.sum(radial_tangent**2, axis=0))
                    / jnp.abs(radial_metric),
                  ray_shift_tangent=jnp.sqrt(jnp.sum(shift_tangent**2, axis=0)))
    for name in ("alpha", "beta", "chi", "conformal_grr", "conformal_gt"):
        result[name] = result["ray_" + name][0]
    return {key: np.asarray(value) for key, value in result.items()}


def read_profiles(directory):
    """Sample radial rays from every native openPMD field snapshot.

    Reads the global C-node meshes the PyPIC3D executable writes and never
    evolves anything. The returned list is ordered by step, matching what the
    comparison functions below expect.
    """
    directory = Path(directory)
    series = io.Series(str(directory / "data" / "fields.h5"), io.Access.read_only)
    scalar = io.Mesh_Record_Component.SCALAR
    profiles = []
    try:
        steps = sorted(series.iterations)
        if not steps:
            raise FileNotFoundError(f"No Cartesian OS field snapshots in {directory}")
        for step in steps:
            iteration = series.iterations[step]
            required = {"lapse", "conformal_factor", "shift"} | {
                f"conformal_metric_{name}" for name in METRIC_COMPONENTS}
            missing = required - set(iteration.meshes)
            if missing:
                raise ValueError(f"Snapshot {step} lacks metric meshes: {sorted(missing)}")
            reference = iteration.meshes["conformal_factor"]
            shape = tuple(int(value) for value in reference[scalar].shape)
            if len(shape) != 3 or min(shape) < 4:
                raise ValueError(f"Expected a resolved 3D metric, got {shape}")
            spacing = np.asarray(reference.grid_spacing) * reference.grid_unit_SI
            origin = np.asarray(reference.grid_global_offset) * reference.grid_unit_SI \
                + np.asarray(reference[scalar].position) * spacing
            if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) \
                    or not np.all(np.isfinite(origin)):
                raise ValueError(f"Invalid metric coordinates at snapshot {step}")
            if np.any(np.abs(spacing - spacing[0]) > 1.e-12 * spacing[0]):
                raise ValueError("The ray sampler needs equal spacing on all axes")

            pending = {"lapse": iteration.meshes["lapse"][scalar].load_chunk(),
                       "conformal_factor": reference[scalar].load_chunk()}
            for axis in "xyz":
                pending["shift_" + axis] = iteration.meshes["shift"][axis].load_chunk()
            for name in METRIC_COMPONENTS:
                pending[name] = iteration.meshes[f"conformal_metric_{name}"][scalar].load_chunk()
            series.flush()

            components = {name: np.asarray(pending[name]) for name in METRIC_COMPONENTS}
            conformal_metric = np.stack([
                np.stack([components["".join(sorted(("xyz"[i], "xyz"[j])))] for j in range(3)])
                for i in range(3)])
            meshes = dict(lapse=np.asarray(pending["lapse"]),
                          conformal_factor=np.asarray(pending["conformal_factor"]),
                          shift=np.stack([np.asarray(pending["shift_" + axis]) for axis in "xyz"]),
                          conformal_metric=conformal_metric)
            if not all(np.all(np.isfinite(value)) for value in meshes.values()):
                raise ValueError(f"Snapshot {step} has a non-finite metric")
            if not np.all(meshes["conformal_factor"] > 0):
                raise ValueError(f"Snapshot {step} has a non-positive conformal factor")
            profile = build_profile(meshes, shape, origin, float(spacing[0]))
            profile["step"] = np.asarray(step)
            profile["time"] = np.asarray(float(iteration.time * iteration.time_unit_SI))
            profiles.append(profile)
    finally:
        series.close()
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


def fit_lapse_normalization(numerical, analytical, window, radius_over_mass):
    """Fit the one asymptotic lapse scale a finite outer boundary leaves behind.

    The stationary Killing vector is shared, so g_tt is chart independent and a
    1+log lapse that has not settled to unity near the boundary enters the
    comparison only as the constant factor ``alpha_infinity**2``. g_RR is
    invariant under that rescaling and is not corrected.

    The scale is asymptotic, so it is fitted on the outer half of the
    comparison window, farthest from the horizon; the inner half carries
    genuine geometric error that would bias the fit. ``ratio_spread`` reports
    whether one constant actually describes the fitted region. Radial
    structure left after the correction is geometric error, not gauge.
    """
    lower = .5*(2.5 + INNER_RADIUS_LIMIT)
    fit = window & (radius_over_mass >= lower)
    ratio = numerical[fit]/analytical[fit]
    if ratio.size < 2 or not np.all(ratio > 0.):
        return None
    scale = float(np.sqrt(np.mean(ratio)))
    return dict(alpha_infinity=scale, relative_offset=scale - 1.,
                ratio_spread=float(np.max(ratio) - np.min(ratio)),
                samples=int(ratio.size), fit_window_over_mass=[lower, INNER_RADIUS_LIMIT],
                definition="alpha_infinity**2 = mean(g_tt numerical / g_tt Schwarzschild) in the fit window",
                applies_to="g_tt only; g_RR is invariant under a lapse rescaling")


def error_history(profiles, mass, vacuum_from=20.):
    """Relative metric error at every saved profile, and its late growth rate.

    After the matter is excised the spacetime is stationary vacuum, so a
    boundary-free evolution would hold this flat. A linear growth rate over
    ``vacuum_from`` onward measures how fast the outer boundary contaminates
    the interior; it is set by the boundary location, not by dx.
    """
    series = []
    for profile in profiles:
        result = reconstruct_metric(profile)
        R, entry = np.asarray(result["R"]), dict(time_over_mass=float(profile["time"])/mass)
        for name in ("g_RR", "g_tt"):
            numerical = np.asarray(result[name])
            with np.errstate(divide="ignore", invalid="ignore"):
                exact = (1./(1.-2*mass/R) if name == "g_RR" else -(1.-2*mass/R))
            window = (np.isfinite(numerical) & np.isfinite(exact)
                      & (R/mass >= 2.5) & (R/mass <= INNER_RADIUS_LIMIT))
            window[-2:] = False
            relative = (numerical[window] - exact[window])/exact[window]
            entry[name] = float(np.sqrt(np.mean(relative**2))) if relative.size else None
        series.append(entry)
    report = dict(series=series, vacuum_from_over_mass=vacuum_from,
                  radius_window_over_mass=[2.5, INNER_RADIUS_LIMIT],
                  note="relative RMS per saved profile; growth rate is a linear fit for t >= vacuum_from")
    for name in ("g_RR", "g_tt"):
        late = [(e["time_over_mass"], e[name]) for e in series
                if e[name] is not None and e["time_over_mass"] >= vacuum_from]
        report[name + "_growth_rate_per_mass"] = (
            float(np.polyfit([t for t, _ in late], [v for _, v in late], 1)[0])
            if len(late) >= 2 else None)
        report[name + "_samples"] = len(late)
    return report


def comparison_report(profiles, mass):
    result = {key: np.asarray(value) for key, value in reconstruct_metric(profiles[-1]).items()}
    R = result["R"]
    analytical = {"g_RR": np.full_like(R, np.nan), "g_tt": np.full_like(R, np.nan)}
    exterior = (R > 2*mass) & (np.arange(R.size) >= int(outward_branch(R)))
    analytical["g_RR"][exterior] = 1./(1.-2*mass/R[exterior])
    analytical["g_tt"][exterior] = -(1.-2*mass/R[exterior])
    assumptions = measure_assumptions(profiles, mass, R)
    report = dict(time_over_mass=float(profiles[-1]["time"])/mass,
                  initial_gravitational_mass=mass, assumptions=assumptions,
                  interpretation=("Stationary spherical overlay supported by these diagnostics"
                                  if assumptions["supported"] else "Provisional stationary spherical overlay"),
                  accuracy_claim="No convergence or boundary-independence claim from this single coarse run",
                  lapse_normalization=None, errors={})
    for name in analytical:
        numerical = result[name]
        valid = np.isfinite(numerical) & np.isfinite(analytical[name])
        valid[-2:] = False
        window = valid & (R/mass >= 2.5) & (R/mass <= INNER_RADIUS_LIMIT)
        relative = (numerical[window] - analytical[name][window])/analytical[name][window]
        metrics = dict(relative_rms=float(np.sqrt(np.mean(relative**2))) if relative.size else None,
                       relative_max=float(np.max(np.abs(relative))) if relative.size else None,
                       radius_window_over_mass=[2.5, INNER_RADIUS_LIMIT], probes={})
        if name == "g_tt":
            # Separate the unfitted gauge constant from geometric error. The raw
            # numbers above are retained; refinement cannot reduce this offset,
            # which the outer boundary location fixes.
            report["lapse_normalization"] = fit_lapse_normalization(
                numerical, analytical[name], window, R/mass)
            if report["lapse_normalization"] is not None:
                scaled = numerical[window]/report["lapse_normalization"]["alpha_infinity"]**2
                corrected = (scaled - analytical[name][window])/analytical[name][window]
                metrics["gauge_removed"] = dict(
                    relative_rms=float(np.sqrt(np.mean(corrected**2))),
                    relative_max=float(np.max(np.abs(corrected))),
                    note="g_tt after dividing out one fitted asymptotic lapse scale")
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
    # Every saved profile, not just the final one: the growth rate over the
    # vacuum phase is what distinguishes boundary error from discretization.
    report["error_history"] = error_history(profiles, mass)
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
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    settings = read_configuration()
    directory = args.input if args.input is not None else settings["output_dir"]
    profiles = read_profiles(directory)
    report = save_comparison(profiles, settings,
                             args.output_directory or Path(directory) / "analytical_comparison")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
