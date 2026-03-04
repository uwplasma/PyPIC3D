#!/usr/bin/env python3
#
# --- Analysis script for 2D magnetic reconnection using OpenPMD field output.

from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys

# Avoid ABI mismatches from user-site wheels (e.g., NumPy) shadowing the
# environment-managed scientific stack used by h5py.
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({"font.size": 20})

# User-defined normalization constants.
# Set to None to keep SI units on the reconnection-rate plot.
T_CI = 1.428954703096425e-07  # ion cyclotron time [s]
V_A = 7.49e+06   # Alfven speed [m/s]
B0 = 0.1    # reference magnetic field [T]


def decode_axis_labels(raw_labels):
    labels = []
    for label in raw_labels:
        if isinstance(label, (bytes, np.bytes_)):
            labels.append(label.decode())
        else:
            labels.append(str(label))
    return labels


def resolve_openpmd_h5(path):
    path = Path(path)
    if path.suffix == ".h5":
        return path
    if path.suffix != ".pmd":
        raise ValueError(f"Expected a .pmd or .h5 path, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"OpenPMD index file not found: {path}")

    target = path.read_text(encoding="utf-8").strip()
    if not target:
        raise ValueError(f"OpenPMD index file is empty: {path}")

    if "%T" in target:
        pattern = target.replace("%T", "*")
        matches = sorted(path.parent.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No HDF5 files match series pattern '{pattern}' from {path}"
            )
        return matches[0]

    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = path.parent / target_path
    return target_path


def get_iteration_keys(h5_file):
    if "data" not in h5_file:
        raise KeyError("OpenPMD file does not contain /data group")
    keys = list(h5_file["data"].keys())
    if not keys:
        raise ValueError("No iterations found in /data group")
    return sorted(
        keys,
        key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)),
    )


def get_mesh_metadata(mesh_group):
    labels = decode_axis_labels(mesh_group.attrs["axisLabels"])
    spacing = np.asarray(mesh_group.attrs["gridSpacing"], dtype=float)
    offset = np.asarray(mesh_group.attrs["gridGlobalOffset"], dtype=float)
    if len(labels) != len(spacing) or len(labels) != len(offset):
        raise ValueError(
            "Inconsistent OpenPMD mesh metadata: axisLabels/gridSpacing/gridGlobalOffset"
        )
    return labels, spacing, offset


def component_to_xz_plane(component_data, axis_labels, spacing, offset):
    data = np.asarray(component_data)
    labels = list(axis_labels)
    spacing_by_label = {lbl: float(spacing[ii]) for ii, lbl in enumerate(labels)}
    offset_by_label = {lbl: float(offset[ii]) for ii, lbl in enumerate(labels)}

    if "y" in labels:
        y_axis = labels.index("y")
        data = np.take(data, data.shape[y_axis] // 2, axis=y_axis)
        labels.pop(y_axis)

    if "x" not in labels or "z" not in labels:
        raise ValueError(f"Expected x/z axes in OpenPMD labels, got: {labels}")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D x-z data after slicing, got shape={data.shape}")

    x_axis = labels.index("x")
    z_axis = labels.index("z")
    if (x_axis, z_axis) != (0, 1):
        data = np.moveaxis(data, (x_axis, z_axis), (0, 1))

    x_grid = offset_by_label["x"] + spacing_by_label["x"] * np.arange(data.shape[0])
    z_grid = offset_by_label["z"] + spacing_by_label["z"] * np.arange(data.shape[1])
    return data, x_grid, z_grid


def load_component_xz(h5_file, iteration_key, record, component):
    mesh_group = h5_file[f"data/{iteration_key}/meshes/{record}"]
    labels, spacing, offset = get_mesh_metadata(mesh_group)
    component_data = np.asarray(mesh_group[component])
    return component_to_xz_plane(component_data, labels, spacing, offset)


def load_frame(h5_file, iteration_key):
    bx, x_grid, z_grid = load_component_xz(h5_file, iteration_key, "B", "x")
    by, _, _ = load_component_xz(h5_file, iteration_key, "B", "y")
    bz, _, _ = load_component_xz(h5_file, iteration_key, "B", "z")
    jy, _, _ = load_component_xz(h5_file, iteration_key, "J", "y")
    ey, _, _ = load_component_xz(h5_file, iteration_key, "E", "y")

    iteration = h5_file[f"data/{iteration_key}"]
    time_value = float(np.asarray(iteration.attrs.get("time", np.nan)).reshape(-1)[0])
    dt_value = float(np.asarray(iteration.attrs.get("dt", np.nan)).reshape(-1)[0])

    return {
        "Bx": bx,
        "By": by,
        "Bz": bz,
        "Jy": jy,
        "Ey": ey,
        "x_grid": x_grid,
        "z_grid": z_grid,
        "time": time_value,
        "dt": dt_value,
    }


def axis_extent(axis_values):
    if axis_values.size <= 1:
        center = float(axis_values[0]) if axis_values.size == 1 else 0.0
        return center - 0.5, center + 0.5
    delta = float(axis_values[1] - axis_values[0])
    return float(axis_values[0] - 0.5 * delta), float(axis_values[-1] + 0.5 * delta)


def trace_field_lines(Bx, Bz, x_grid, z_grid, n_lines=10, max_steps=5000):
    x_min, x_max = float(np.min(x_grid)), float(np.max(x_grid))
    z_min, z_max = float(np.min(z_grid)), float(np.max(z_grid))

    step_x = abs(float(x_grid[1] - x_grid[0])) if x_grid.size > 1 else 1.0
    step_z = abs(float(z_grid[1] - z_grid[0])) if z_grid.size > 1 else 1.0
    step_size = 0.25 * min(step_x, step_z)

    bx_interp = RegularGridInterpolator(
        (x_grid, z_grid), Bx, bounds_error=False, fill_value=np.nan
    )
    bz_interp = RegularGridInterpolator(
        (x_grid, z_grid), Bz, bounds_error=False, fill_value=np.nan
    )

    start_x = np.full(n_lines, x_min)
    start_x[: n_lines // 2] = x_max
    start_z = np.linspace(z_min * 0.9, z_max * 0.9, n_lines)

    paths = []
    for sx, sz in zip(start_x, start_z):
        path_x = [float(sx)]
        path_z = [float(sz)]

        for _ in range(max_steps):
            point = np.array([path_x[-1], path_z[-1]])
            b_x = float(np.asarray(bx_interp(point)).reshape(-1)[0])
            b_z = float(np.asarray(bz_interp(point)).reshape(-1)[0])

            if not np.isfinite(b_x) or not np.isfinite(b_z):
                break
            b_mag = float(np.hypot(b_x, b_z))
            if b_mag == 0.0:
                break

            x_new = path_x[-1] + step_size * b_x / b_mag
            z_new = path_z[-1] + step_size * b_z / b_mag

            if (x_new < x_min) or (x_new > x_max) or (z_new < z_min) or (z_new > z_max):
                break

            path_x.append(x_new)
            path_z.append(z_new)

        paths.append((path_x, path_z))
    return paths


def compute_reconnection_series(h5_file, iteration_keys):
    times = []
    ey_mean = []
    for key in iteration_keys:
        frame = load_frame(h5_file, key)
        times.append(frame["time"])
        ey_mean.append(np.mean(frame["Ey"]))
    return np.asarray(times, dtype=float), np.asarray(ey_mean, dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze magnetic reconnection from OpenPMD HDF5 field output."
    )
    parser.add_argument(
        "--fields",
        default="data/fields.pmd",
        help="Path to OpenPMD field index (.pmd) or HDF5 file (.h5).",
    )
    parser.add_argument(
        "--output-dir",
        default="diags",
        help="Directory to write analysis plots and animation.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Only generate reconnection-rate plot.",
    )
    parser.add_argument("--fps", type=int, default=6, help="Animation frame rate.")
    return parser.parse_args()


def main():
    args = parse_args()

    fields_h5 = resolve_openpmd_h5(args.fields)
    if not fields_h5.exists():
        raise FileNotFoundError(
            f"OpenPMD HDF5 file not found: {fields_h5}. "
            f"Check that the simulation wrote field output."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(fields_h5, "r") as h5_file:
        iteration_keys = get_iteration_keys(h5_file)
        # The final write can be incomplete if the simulation stopped mid-output.
        iteration_keys = iteration_keys[:-1]
        num_steps = len(iteration_keys)
        if num_steps == 0:
            raise ValueError(
                "No complete OpenPMD iterations found after dropping the last write."
            )

        times, ey_mean = compute_reconnection_series(h5_file, iteration_keys)

        if not np.all(np.isfinite(times)):
            # fall back to dt metadata if explicit time was not written
            first = load_frame(h5_file, iteration_keys[0])
            dt = first["dt"] if np.isfinite(first["dt"]) else 1.0
            times = np.arange(num_steps, dtype=float) * dt

        if T_CI is not None and T_CI != 0:
            time_plot = times / T_CI
            xlabel = r"$t/\tau_{c,i}$"
        else:
            time_plot = times
            xlabel = "t [s]"

        if V_A is not None and B0 is not None and (V_A * B0) != 0:
            ey_plot = ey_mean / (V_A * B0)
            ylabel = r"$\langle E_y \rangle/(v_A B_0)$"
        else:
            ey_plot = ey_mean
            ylabel = r"$\langle E_y \rangle$"

        plt.figure()
        plt.plot(time_plot, ey_plot, "o-")
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Reconnection rate")
        plt.tight_layout()
        plt.savefig(output_dir / "reconnection_rate.png")
        plt.close()

        if args.skip_animation:
            return

        from matplotlib.animation import FFMpegWriter, FuncAnimation

        frame0 = load_frame(h5_file, iteration_keys[0])
        x_grid = frame0["x_grid"]
        z_grid = frame0["z_grid"]

        x_lo, x_hi = axis_extent(x_grid)
        z_lo, z_hi = axis_extent(z_grid)
        extent = [x_lo, x_hi, z_lo, z_hi]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_ylabel("$z$")
        axes[2].set_xlabel("$x$")

        jy_abs = np.max(np.abs(frame0["Jy"]))
        if jy_abs == 0:
            jy_abs = 1.0

        sX = axes[0].imshow(
            frame0["Jy"].T,
            origin="lower",
            norm=colors.TwoSlopeNorm(vmin=-jy_abs, vcenter=0.0, vmax=jy_abs),
            extent=extent,
            cmap=plt.cm.RdYlBu_r,
        )
        plt.colorbar(sX, ax=axes[0], label="$J_y$")

        sY = axes[1].imshow(
            frame0["By"].T,
            origin="lower",
            extent=extent,
            cmap=plt.cm.plasma,
        )
        plt.colorbar(sY, ax=axes[1], label="$B_y$")

        bz_abs = np.max(np.abs(frame0["Bz"]))
        if bz_abs == 0:
            bz_abs = 1.0

        sZ = axes[2].imshow(
            frame0["Bz"].T,
            origin="lower",
            extent=extent,
            norm=colors.TwoSlopeNorm(vmin=-bz_abs, vcenter=0.0, vmax=bz_abs),
            cmap=plt.cm.RdBu,
        )
        plt.colorbar(sZ, ax=axes[2], label="$B_z$")

        field_line_paths = trace_field_lines(frame0["Bx"], frame0["Bz"], x_grid, z_grid)
        field_lines = []
        for path_x, path_z in field_line_paths:
            (line,) = axes[2].plot(path_x, path_z, "--", color="k")
            field_lines.append(line)

        title = axes[0].set_title(f"Iteration {iteration_keys[0]}")

        def animate(frame_idx):
            frame = load_frame(h5_file, iteration_keys[frame_idx])
            sX.set_array(frame["Jy"].T)
            sY.set_array(frame["By"].T)
            sZ.set_array(frame["Bz"].T)

            bz_scale = np.max(np.abs(frame["Bz"]))
            if bz_scale > 0:
                sZ.set_clim(-bz_scale, bz_scale)

            new_paths = trace_field_lines(frame["Bx"], frame["Bz"], x_grid, z_grid)
            for line, (path_x, path_z) in zip(field_lines, new_paths):
                line.set_data(path_x, path_z)

            title.set_text(f"Iteration {iteration_keys[frame_idx]}")
            return [sX, sY, sZ, *field_lines, title]

        anim = FuncAnimation(fig, animate, frames=num_steps, repeat=True, blit=False)
        try:
            writer = FFMpegWriter(fps=args.fps)
            anim.save(output_dir / "mag_reconnection.mp4", writer=writer)
        except FileNotFoundError:
            from matplotlib.animation import PillowWriter

            gif_writer = PillowWriter(fps=args.fps)
            anim.save(output_dir / "mag_reconnection.gif", writer=gif_writer)
        plt.close(fig)


if __name__ == "__main__":
    main()
