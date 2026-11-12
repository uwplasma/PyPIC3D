"""Plot the Cartesian OS central and minimum lapse histories.

Adapted from RadiShPICR/demos/os_collapse_Z4C_GR/plot_lapse_history.py.
The initial lapse is unity here, as in Wu and Most, arXiv:2611.12831v1.
"""

import argparse
import csv
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    from diagnostics import os_lapse_reference
else:
    from .diagnostics import os_lapse_reference


DEFAULT_INPUT = Path(__file__).with_name("outputs")


def read_lapse_history(directory):
    """Read every accepted step and check the saved central proper clock."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    with (directory / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    history = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    time, alpha = history["time"], history["central_lapse"]
    if time[0] != 0. or np.any(np.diff(time) <= 0.) or not np.all(np.isfinite(alpha)):
        raise ValueError("Central proper time needs finite lapse, t=0, and increasing saved times")
    tau = np.zeros_like(time)
    tau[1:] = np.cumsum(.5*(alpha[:-1] + alpha[1:])*np.diff(time))
    if not np.allclose(history["central_proper_time"], tau, rtol=1.e-10, atol=1.e-12):
        raise ValueError("Saved central proper time disagrees with the accepted-step lapse integral")
    return history, metadata


def save_lapse_history(history, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mass = metadata["total_mass"]
    reference_time, reference_lapse = os_lapse_reference(mass, metadata["surface_areal_radius"])
    clock_supported = ("central_shift_over_lapse" not in history
                       or np.max(history["central_shift_over_lapse"]) < .01)
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
            axis.axhline(metadata["lapse_threshold"], color="gray", linestyle=":",
                         label="Particle deletion threshold")
            note = "Minimum over all physical Cartesian metric nodes."
        axis.set(xlabel=xlabel, ylabel=ylabel,
                 title=rf"Neutral OS collapse ($R_0/M={metadata['surface_areal_radius']/mass:g}$)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
        figure.text(.5, .06, note, ha="center", fontsize=8)
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"{name}.{extension}", dpi=180)
        plt.close(figure)

    early = history["central_proper_time"]/mass <= 2.
    analytic = np.interp(history["central_proper_time"][early]/mass,
                         np.asarray(reference_time), np.asarray(reference_lapse))
    result = dict(final_time_over_mass=float(history["time"][-1]/mass),
                  final_central_proper_time_over_mass=float(history["central_proper_time"][-1]/mass),
                  final_central_lapse=float(history["central_lapse"][-1]),
                  final_minimum_lapse=float(history["minimum_lapse"][-1]),
                  initial_central_lapse=float(history["central_lapse"][0]),
                  early_window="tau_c/M <= 2; absolute lapse error",
                  early_central_lapse_max_absolute_error=float(np.max(np.abs(
                      history["central_lapse"][early] - analytic))),
                  early_window_complete=bool(history["central_proper_time"][-1]/mass >= 2.),
                  final_active_particles=int(history["active_particles"][-1]),
                  final_retained_rest_mass=float(history["retained_rest_mass"][-1]),
                  particles_deleted_by_lapse=int(history["deleted_lapse"][-1]),
                  particles_escaped_outer=int(history["escaped_outer"][-1]))
    if "central_shift_over_lapse" in history:
        result["maximum_central_shift_over_lapse"] = float(np.max(history["central_shift_over_lapse"]))
        result["central_clock_assumption_supported"] = result["maximum_central_shift_over_lapse"] < .01
    (directory / "lapse_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    history, metadata = read_lapse_history(args.input)
    result = save_lapse_history(history, metadata, args.output_directory or args.input / "lapse_history")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
