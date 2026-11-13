"""Plot the Cartesian OS central and minimum lapse histories from openPMD.

Reads the native ``data/fields.h5`` and ``data/particles.h5`` written by the
PyPIC3D executable; it never evolves anything. The initial lapse is unity here,
as in Wu and Most, arXiv:2611.12831v1.

The central proper clock is the trapezoid integral of the central lapse over the
*written* snapshots, so its accuracy is set by ``[plotting] plotting_interval``
in os_collapse.toml. At interval 1 it samples every macro step.

    python plot_central_lapse.py [run-directory]
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import matplotlib
import numpy as np
import openpmd_api as io
import toml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    from diagnostics import os_lapse_reference
    from initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS
else:
    from .diagnostics import os_lapse_reference
    from .initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS


CONFIG = Path(__file__).with_name("os_collapse.toml")
SHIFT_TOLERANCE = .01


def read_configuration(path=CONFIG):
    """Take the run directory and deletion threshold from the TOML itself."""
    config = toml.load(path)
    parameters = {**config.get("simulation_parameters", {}),
                  **config.get("static_parameters", {})}
    return dict(output_dir=Path(parameters.get("output_dir", ".")),
                lapse_threshold=float(parameters.get("lapse_threshold", .1)),
                total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS)


def central_index(mesh, component):
    """Index of the coordinate origin, which must be an actual C node."""
    spacing = np.asarray(mesh.grid_spacing) * mesh.grid_unit_SI
    origin = np.asarray(mesh.grid_global_offset) * mesh.grid_unit_SI \
        + np.asarray(component.position) * spacing
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Invalid metric grid spacing")
    exact = -origin / spacing
    index = np.rint(exact).astype(int)
    if np.any(np.abs(exact - index) > 1.e-9) or np.any(index < 0) \
            or np.any(index >= np.asarray(component.shape)):
        raise ValueError("The coordinate origin is not a grid node; "
                         "the central lapse would need interpolation")
    return tuple(int(value) for value in index)


def read_lapse_history(directory):
    """Collect per-snapshot central/minimum lapse, shift ratio and particle totals."""
    directory = Path(directory)
    fields = io.Series(str(directory / "data" / "fields.h5"), io.Access.read_only)
    particles = io.Series(str(directory / "data" / "particles.h5"), io.Access.read_only)
    scalar = io.Mesh_Record_Component.SCALAR
    rows = []
    try:
        steps = sorted(fields.iterations)
        if not steps:
            raise FileNotFoundError(f"No field snapshots in {directory}")
        for step in steps:
            iteration = fields.iterations[step]
            if "lapse" not in iteration.meshes:
                raise ValueError(f"Snapshot {step} has no lapse mesh")
            mesh = iteration.meshes["lapse"]
            center = central_index(mesh, mesh[scalar])
            pending = {"lapse": mesh[scalar].load_chunk()}
            for axis in "xyz":
                pending["shift_" + axis] = iteration.meshes["shift"][axis].load_chunk()
            fields.flush()
            alpha = np.asarray(pending["lapse"])
            shift = np.stack([np.asarray(pending["shift_" + axis])[center] for axis in "xyz"])
            if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(shift)):
                raise ValueError(f"Snapshot {step} has a non-finite lapse or shift")
            central = float(alpha[center])
            if central <= 0.:
                raise ValueError(f"Snapshot {step} has a non-positive central lapse")
            row = dict(step=step, time=float(iteration.time * iteration.time_unit_SI),
                       central_lapse=central, minimum_lapse=float(np.min(alpha)),
                       central_shift_over_lapse=float(np.linalg.norm(shift) / central))

            # Survivors only: openPMD records the particles still active, so the
            # loss cannot be split into lapse deletions and outer escapes here.
            species = particles.iterations[step].particles["dust"]
            weighting = species["weighting"][scalar]
            mass = species["mass"][scalar]
            pending_particles = (weighting.load_chunk(), mass.load_chunk())
            particles.flush()
            weights, masses = (np.asarray(value) for value in pending_particles)
            row["active_particles"] = int(weights.size)
            row["retained_rest_mass"] = float(np.sum(weights * masses))
            rows.append(row)
    finally:
        for series in (particles, fields):
            series.close()

    history = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    time, alpha = history["time"], history["central_lapse"]
    if time[0] != 0. or np.any(np.diff(time) <= 0.):
        raise ValueError("Central proper time needs t=0 and increasing snapshot times")
    tau = np.zeros_like(time)
    tau[1:] = np.cumsum(.5 * (alpha[:-1] + alpha[1:]) * np.diff(time))
    history["central_proper_time"] = tau
    return history


def save_lapse_history(history, settings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mass = settings["total_mass"]
    reference_time, reference_lapse = os_lapse_reference(mass, settings["surface_areal_radius"])
    clock_supported = np.max(history["central_shift_over_lapse"]) < SHIFT_TOLERANCE
    for name, time_key, lapse_key, xlabel, ylabel in (
        ("central_lapse_proper_time", "central_proper_time", "central_lapse",
         r"Central proper time $\tau_c/M$", r"Central lapse $\alpha_c$"),
        ("minimum_lapse_coordinate_time", "time", "minimum_lapse",
         r"Coordinate time $t/M$", r"Minimum lapse $\min\alpha$"),
    ):
        figure, axis = plt.subplots(figsize=(7.6, 5.))
        figure.subplots_adjust(left=.12, right=.97, top=.88, bottom=.23)
        axis.plot(history[time_key]/mass, history[lapse_key], label="Cartesian PIC/BSSN",
                  color="tab:blue", linewidth=1.8)
        if lapse_key == "central_lapse":
            axis.plot(reference_time, reference_lapse, ":", color="black", linewidth=1.8,
                      label=r"Unit-initial-lapse OS limit $1+6\ln(R/R_0)$")
            note = "Reference applies to the homogeneous interior before surface gauge modes arrive."
            if not clock_supported:
                xlabel = r"Central lapse integral $\tau_c=\int\alpha_c\,dt$ ($M=1$)"
                note += "\nLate central shift is non-negligible; this is not a fixed-center proper clock."
        else:
            axis.axhline(settings["lapse_threshold"], color="gray", linestyle=":",
                         label="Particle deletion threshold")
            note = "Minimum over all physical Cartesian metric nodes."
        axis.set(xlabel=xlabel, ylabel=ylabel,
                 title=rf"Neutral OS collapse ($R_0/M={settings['surface_areal_radius']/mass:g}$)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
        figure.text(.5, .06, note, ha="center", fontsize=8)
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"{name}.{extension}", dpi=180)
        plt.close(figure)

    early = history["central_proper_time"]/mass <= 2.
    analytic = np.interp(history["central_proper_time"][early]/mass,
                         np.asarray(reference_time), np.asarray(reference_lapse))
    intervals = np.diff(history["time"])
    result = dict(final_time_over_mass=float(history["time"][-1]/mass),
                  final_central_proper_time_over_mass=float(history["central_proper_time"][-1]/mass),
                  final_central_lapse=float(history["central_lapse"][-1]),
                  final_minimum_lapse=float(history["minimum_lapse"][-1]),
                  initial_central_lapse=float(history["central_lapse"][0]),
                  snapshots=int(history["time"].size),
                  snapshot_interval_over_mass=float(np.max(intervals)/mass) if intervals.size else 0.,
                  early_window="tau_c/M <= 2; absolute lapse error",
                  early_central_lapse_max_absolute_error=float(np.max(np.abs(
                      history["central_lapse"][early] - analytic))) if np.any(early) else float("nan"),
                  early_window_complete=bool(history["central_proper_time"][-1]/mass >= 2.),
                  initial_active_particles=int(history["active_particles"][0]),
                  final_active_particles=int(history["active_particles"][-1]),
                  initial_rest_mass=float(history["retained_rest_mass"][0]),
                  final_retained_rest_mass=float(history["retained_rest_mass"][-1]),
                  # openPMD stores survivors, so lapse deletions and outer escapes
                  # cannot be separated from these files.
                  particles_removed=int(history["active_particles"][0]
                                        - history["active_particles"][-1]),
                  maximum_central_shift_over_lapse=float(np.max(history["central_shift_over_lapse"])),
                  central_clock_assumption_supported=bool(clock_supported))
    if not math.isfinite(result["early_central_lapse_max_absolute_error"]):
        del result["early_central_lapse_max_absolute_error"]
    (directory / "lapse_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    settings = read_configuration()
    directory = args.input if args.input is not None else settings["output_dir"]
    history = read_lapse_history(directory)
    result = save_lapse_history(history, settings,
                                args.output_directory or Path(directory) / "lapse_history")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
