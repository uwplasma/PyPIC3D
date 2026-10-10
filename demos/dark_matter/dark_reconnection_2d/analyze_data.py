"""Compare dark, ordinary and particle-force magnetic fields from a saved run."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation, colors
import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io
import toml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demos.dark_matter.dark_reconnection_2d.initial_data import harris_scales
from demos.standard_yee.reconnection_2d.analyze_data import resolve_series_path


def read_frame(series, index, mixing):
    """Collocate magnetic components using openPMD positions, without smoothing.

    x interpolation wraps periodically. z stops before the uppermost Bz node,
    so no values are extrapolated through a conducting wall. Flux is sampled
    from the unsmoothed Ay mesh at the initial O/X reference locations.
    """
    iteration = series.iterations[index]
    mesh = iteration.meshes["B"]
    shape = tuple(mesh["x"].shape)
    if list(mesh.axis_labels) != ["x", "y", "z"] or len(shape) != 3 or shape[1] != 1:
        raise ValueError("Expected an x-z mesh with Ny=1")
    spacing = np.asarray(mesh.grid_spacing) * mesh.grid_unit_SI
    offset = np.asarray(mesh.grid_global_offset) * mesh.grid_unit_SI
    x = offset[0] + (np.arange(shape[0]) + .5) * spacing[0]
    z = offset[2] + (np.arange(shape[2]-1) + .5) * spacing[2]
    length = shape[0] * spacing[0]
    pending = {}
    for name, components in (("B", "xyz"), ("dark_B", "xyz"), ("dark_A", "y")):
        record = iteration.meshes[name]
        if (list(record.axis_labels) != ["x", "y", "z"]
                or not np.allclose(np.asarray(record.grid_spacing)*record.grid_unit_SI, spacing, rtol=1e-12, atol=0)
                or not np.allclose(np.asarray(record.grid_global_offset)*record.grid_unit_SI, offset, rtol=1e-12, atol=0)):
            raise ValueError("Ordinary and dark meshes must share the same physical grid")
        for component in components:
            pending[name, component] = record[component].load_chunk()
    series.flush()

    magnetic = {"B": [], "dark_B": []}
    for (name, component), values in pending.items():
        record = iteration.meshes[name][component]
        if tuple(values.shape) != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid shape or nonfinite data at iteration {index}: {name}/{component}")
        values = values[:, 0, :] * record.unit_SI
        position = np.asarray(record.position)
        if name == "B" and position.size == 1:
            # The legacy ordinary writer omits component positions. Both
            # sectors use the same Yee B faces; the dark writer records them.
            position = np.asarray(iteration.meshes["dark_B"][component].position)
        source_x = offset[0] + (np.arange(shape[0]) + position[0]) * spacing[0]
        source_z = offset[2] + (np.arange(shape[2]) + position[2]) * spacing[2]
        if name == "dark_A":
            midplane = np.array([np.interp(0., source_z, row) for row in values])
            flux = (np.interp(-length/2, source_x, midplane, period=length)
                    - np.interp(0., source_x, midplane, period=length))
            continue
        on_z = np.array([np.interp(z, source_z, row) for row in values])
        collocated = np.array([np.interp(x, source_x, column, period=length) for column in on_z.T]).T
        magnetic[name].append(collocated)
    magnetic = {name: np.asarray(values) for name, values in magnetic.items()}
    magnetic["effective_B"] = magnetic["B"] - mixing * magnetic["dark_B"]
    return iteration.time * iteration.time_unit_SI, x, z, magnetic, flux


def draw_field_lines(figure, axes, frame, scales, limits, skin_depth, omega_c):
    time, x, z, magnetic, _ = frame
    titles = (r"Dark $B'$", r"Ordinary $B_{\rm EM}$", r"Particle force $B_{\rm EM}-sB'$")
    for axis, name, title in zip(axes, ("dark_B", "B", "effective_B"), titles):
        axis.clear()
        bx, by, bz = magnetic[name] / scales[name]
        magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        norm = colors.Normalize(0., limits[name])
        axis.pcolormesh(x/skin_depth, z/skin_depth, magnitude.T, shading="auto", cmap="magma", norm=norm)
        if np.any(bx != 0.) or np.any(bz != 0.):
            axis.streamplot(x/skin_depth, z/skin_depth, bx.T, bz.T, color="white",
                            density=1.1, linewidth=.55, arrowsize=.7)
        axis.set(xlabel=r"$x/d_e$", ylabel=r"$z/d_e$", title=title, aspect="equal")
        axis.set_xlim(x[0]/skin_depth, x[-1]/skin_depth)
        axis.set_ylim(z[0]/skin_depth, z[-1]/skin_depth)
    figure.suptitle(rf"Harris-loaded dark sheet: $\Omega_c t={omega_c*time:.4f}$")


def analyze(config, fields_path, output_dir, *, fps=10, dpi=120, movie=True):
    simulation = config["simulation_parameters"]
    skin_depth, b0, _, _, _ = harris_scales(config)
    omega_c = abs(config["particle1"]["charge"]) * b0 / config["particle1"]["mass"]
    mixing = simulation["sin_chi"]
    scales = {"dark_B": b0/abs(mixing), "B": b0, "effective_B": b0}
    limits = dict.fromkeys(scales, 0.)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = io.Series(str(resolve_series_path(fields_path)), io.Access.read_only)
    try:
        indices = sorted(series.iterations)
        if not indices:
            raise ValueError("The field series contains no saved frames")
        times, fluxes = [], []
        for index in indices:
            time, _, _, magnetic, flux = read_frame(series, index, mixing)
            times.append(time)
            fluxes.append(flux)
            for name in scales:
                magnitude = np.sqrt(np.sum(magnetic[name]**2, axis=0)) / scales[name]
                limits[name] = max(limits[name], float(magnitude.max()))
        limits = {name: 1.02*value if value > 0. else 1. for name, value in limits.items()}

        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for axis, name in zip(axes, scales):
            bar = figure.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, limits[name]), cmap="magma"), ax=axis)
            bar.set_label(r"$|B'|/(B_0/|s|)$" if name == "dark_B" else r"$|B|/B_0$")
        try:
            for index, name in ((indices[0], "initial_fields.png"), (indices[-1], "final_fields.png")):
                draw_field_lines(figure, axes, read_frame(series, index, mixing), scales, limits, skin_depth, omega_c)
                figure.savefig(output_dir/name, dpi=dpi)
            if movie:
                writer = animation.FFMpegWriter(fps=fps, codec="h264", bitrate=3000)
                with writer.saving(figure, str(output_dir/"field_lines.mp4"), dpi=dpi):
                    for index in indices:
                        draw_field_lines(figure, axes, read_frame(series, index, mixing), scales, limits, skin_depth, omega_c)
                        writer.grab_frame()
        finally:
            plt.close(figure)
    finally:
        series.close()

    times, fluxes = np.asarray(times), np.asarray(fluxes)
    flux_change = (fluxes-fluxes[0])/(scales["dark_B"]*skin_depth)
    np.savez(output_dir/"flux_history.npz", time=times, dark_flux=fluxes, normalized_flux_change=flux_change)
    energy = np.loadtxt(Path(simulation.get("output_dir", "."))/"data/total_energy.txt", delimiter=",", ndmin=2)
    if not np.all(np.isfinite(energy)):
        raise ValueError("Nonfinite data in total_energy.txt")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(times*omega_c, flux_change)
    axes[0].set(xlabel=r"$\Omega_c t$", ylabel=r"$\Delta\psi'/(B_0 d_e/|s|)$",
                title="Dark flux: initial O/X reference locations")
    axes[1].plot(energy[:, 0]*omega_c, energy[:, 1]/energy[0, 1]-1.)
    axes[1].set(xlabel=r"$\Omega_c t$", ylabel=r"$U(t)/U(0)-1$", title="Total energy, including Proca mass")
    figure.savefig(output_dir/"flux_and_energy.png", dpi=dpi)
    plt.close(figure)
    print(f"Analyzed {len(indices)} saved frames; wrote {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("dark_harris.toml"),
                        help="Default: dark_harris.toml beside this script")
    parser.add_argument("--fields", type=Path, help="Override the run's data/fields.pmd")
    parser.add_argument("--output-dir", type=Path, help="Default: run directory / analysis")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--no-movie", action="store_true", help="Write figures and flux history only")
    args = parser.parse_args(argv)
    config = toml.load(args.config)
    run_dir = Path(config["simulation_parameters"].get("output_dir", "."))
    analyze(config, args.fields or run_dir/"data/fields.pmd", args.output_dir or run_dir/"analysis",
            fps=args.fps, dpi=args.dpi, movie=not args.no_movie)


if __name__ == "__main__":
    main()
