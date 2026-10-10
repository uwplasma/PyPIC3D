# Generate a movie of the magnetic field lines in the reconnection_2d demo.

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import numpy as np
import openpmd_api as io
from demos.standard_yee.reconnection_2d.parameters import B0, CYCLOTRON_FREQUENCY, SKIN_DEPTH


SMOOTHING_SIGMA_CELLS = 2.0


def resolve_series_path(path):
    """Resolve a series path or PyPIC3D's one-line .pmd pointer."""
    path = Path(path).expanduser()
    if path.suffix == ".pmd":
        targets = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(targets) != 1:
            raise ValueError(f"Expected one series path in {path}.")
        target = Path(targets[0]).expanduser()
        path = target if target.is_absolute() else path.parent / target
    return path.resolve()


def read_magnetic_frame(series, index):
    """Return time, x/z coordinates, and B components in SI units."""
    iteration = series.iterations[index]
    if "B" not in iteration.meshes:
        raise ValueError(f"Iteration {index} has no B mesh; enable plot_openpmd_fields.")
    mesh = iteration.meshes["B"]
    if any(component not in mesh for component in "xyz"):
        raise ValueError(f"Iteration {index} requires B/x, B/y, and B/z.")
    shape = tuple(mesh["x"].shape)
    if list(mesh.axis_labels) != ["x", "y", "z"] or len(shape) != 3 or shape[1] != 1:
        raise ValueError(f"Expected an x-y-z mesh with singleton y; got {mesh.axis_labels}, {shape}.")
    if shape[0] < 2 or shape[2] < 3 or any(tuple(mesh[c].shape) != shape for c in "xyz"):
        raise ValueError("Magnetic components must share an (nx, 1, nz) shape with nx >= 2, nz >= 3.")

    pending = [mesh[c].load_chunk() for c in "xyz"]
    series.flush()  # openPMD fills the requested arrays only after flushing.
    magnetic = tuple(
        values[:, 0, :] * mesh[c].unit_SI
        for c, values in zip("xyz", pending)
    )
    if any(not np.all(np.isfinite(values)) for values in magnetic):
        raise ValueError(f"Iteration {index} contains non-finite magnetic fields.")

    spacing = np.asarray(mesh.grid_spacing) * mesh.grid_unit_SI
    offset = np.asarray(mesh.grid_global_offset) * mesh.grid_unit_SI
    if (not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.all(np.isfinite(offset))):
        raise ValueError("Mesh coordinates require finite offsets and positive finite spacings.")
    x = offset[0] + np.arange(shape[0]) * spacing[0]
    z = offset[2] + np.arange(shape[2]) * spacing[2]
    return iteration.time * iteration.time_unit_SI, x, z, magnetic


def smooth_component(values):
    """Gaussian display filter: wrap x and extend the nearest z-edge value."""
    radius = int(np.ceil(4 * SMOOTHING_SIGMA_CELLS))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / SMOOTHING_SIGMA_CELLS) ** 2)
    kernel /= kernel.sum()
    filtered_x = np.zeros_like(values, dtype=float)
    for offset, weight in zip(offsets, kernel):
        filtered_x += weight * np.roll(values, int(offset), axis=0)
    padded = np.pad(filtered_x, ((0, 0), (radius, radius)), mode="edge")
    filtered = np.zeros_like(filtered_x)
    for index, weight in enumerate(kernel):
        filtered += weight * padded[:, index:index + values.shape[1]]
    return filtered


def field_line_quantities(x, z, magnetic):
    """Collocate Yee fields at cell centers and normalize the display arrays."""
    bx, by, bz = magnetic
    bx_center = (0.5 * (bx + np.roll(bx, -1, axis=0)))[:, :-1]
    bz_center = 0.5 * (bz[:, :-1] + bz[:, 1:])
    bx, by, bz = (smooth_component(b) / B0 for b in (bx_center, by[:, :-1], bz_center))
    x = (x + 0.5 * (x[1] - x[0])) / SKIN_DEPTH
    z = (z[:-1] + 0.5 * (z[1] - z[0])) / SKIN_DEPTH
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    return x, z, bx, bz, magnitude


def scan_color_limit(series, indices):
    """Read one frame at a time to find a fixed magnetic color scale."""
    maximum = 0.0
    for index in indices:
        _, x, z, magnetic = read_magnetic_frame(series, index)
        magnitude = field_line_quantities(x, z, magnetic)[-1]
        maximum = max(maximum, float(magnitude.max()))
        del magnetic, magnitude
    if maximum <= 0:
        raise ValueError("No nonzero magnetic field was found in the series.")
    return 1.02 * maximum


def write_movie(series, indices, output_path, color_limit, fps, dpi):
    """Render full-domain field lines with one fixed colorbar."""
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    norm = colors.Normalize(vmin=0, vmax=color_limit)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=axis
    )
    colorbar.set_label(r"$|B|/B_0$")
    writer = animation.FFMpegWriter(fps=fps, codec="h264", bitrate=2400)
    try:
        with writer.saving(figure, str(output_path), dpi=dpi):
            for index in indices:
                time, x, z, magnetic = read_magnetic_frame(series, index)
                x, z, bx, bz, magnitude = field_line_quantities(x, z, magnetic)
                axis.clear()
                axis.set_facecolor("black")
                axis.streamplot(
                    x, z, bx.T, bz.T, color=magnitude.T, cmap="plasma", norm=norm,
                    density=(1.5, 1.2), linewidth=0.9, arrowsize=0.7, broken_streamlines=False,
                )

                axis.set(
                    xlabel=r"$x/d_e$", ylabel=r"$z/d_e$",
                    xlim=(x[0], x[-1]), ylim=(z[0], z[-1]),
                )
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(
                    rf"Magnetic field lines  $\Omega_c t$ = {CYCLOTRON_FREQUENCY * time:.3f}"
                )
                writer.grab_frame()
                del magnetic, bx, bz, magnitude
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=Path("data/fields.pmd"), help="openPMD series or .pmd pointer")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("--fps and --dpi must be positive")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required to write the movie; make ffmpeg available on PATH.")

    path = resolve_series_path(args.fields)
    series = io.Series(str(path), io.Access.read_only)
    try:
        indices = sorted(series.iterations)
        if not indices:
            raise ValueError(f"The field series {path} contains no iterations.")
        print(f"Reading {len(indices)} frame(s) from {path}")
        color_limit = scan_color_limit(series, indices)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "field_lines.mp4"
        write_movie(series, indices, output_path, color_limit, args.fps, args.dpi)
    finally:
        series.close()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
