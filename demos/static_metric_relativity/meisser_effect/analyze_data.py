#!/usr/bin/env python3
"""Compare the numerical Wald run with the analytical solution and make a movie."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpic-wald-demo-matplotlib")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

import wald_solution as wald

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_state import densitize_vector


plt.rcParams.update({"font.size": 12})


def load_scalar(mesh, series):
    """Load one scalar openPMD mesh into an owned NumPy array."""

    record = mesh[io.Mesh_Record_Component.SCALAR]
    pending = record.load_chunk()
    series.flush()
    return np.array(pending, copy=True)


def load_vector(mesh, series):
    """Load the r, theta, and phi records stored under x, y, and z."""

    pending = tuple(mesh[component].load_chunk() for component in ("x", "y", "z"))
    series.flush()
    return tuple(np.array(component, copy=True) for component in pending)


def read_field_series(path):
    """Read ordered times, parallel fields, and magnetic fields from openPMD."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run {path} does not exist. Run run_wald.py first.")

    series = io.Series(str(path), io.Access.read_only)
    times = []
    parallel_fields = []
    magnetic_fields = []

    for iteration_index in series.iterations:
        iteration = series.iterations[iteration_index]
        times.append(float(iteration.time))
        parallel_fields.append(load_scalar(iteration.meshes["E_parallel"], series))
        magnetic_fields.append(load_vector(iteration.meshes["B"], series))

    series.close()
    order = np.argsort(times)
    times = np.asarray(times)[order]
    parallel_fields = np.stack(parallel_fields)[order]
    magnetic_fields = tuple(
        np.stack([magnetic_fields[index][component] for index in order])
        for component in range(3)
    )
    return times, parallel_fields, magnetic_fields


def cell_edges(axis):
    """Return cell edges for one uniformly spaced coordinate-center axis."""

    spacing = axis[1] - axis[0]
    return np.concatenate(
        (
            [axis[0] - 0.5 * spacing],
            0.5 * (axis[:-1] + axis[1:]),
            [axis[-1] + 0.5 * spacing],
        )
    )


def restore_tiled_magnetic_field(B, static_parameters):
    """Restore the one-tile guard-cell layout removed by openPMD output."""

    guard_cells = int(static_parameters.guard_cells)
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    tiled_shape = (1, 1, 1) + tuple(
        width + 2 * guard_cells for width in tile_shape
    )
    interior = (
        0,
        0,
        0,
        slice(guard_cells, -guard_cells),
        slice(guard_cells, -guard_cells),
        slice(guard_cells, -guard_cells),
    )
    B_tiled = tuple(
        jnp.zeros(tiled_shape, dtype=component.dtype).at[interior].set(component)
        for component in B
    )
    return update_tiled_vector_ghost_cells(
        B_tiled,
        static_parameters,
        num_guard_cells=guard_cells,
    )


def center_saved_magnetic_field(B, static_parameters, metric):
    """Apply the production density Yee interpolation to saved physical B."""

    B_tiled = densitize_vector(restore_tiled_magnetic_field(B, static_parameters), metric.B)
    B_center = wald.center_vector(B_tiled, B_FIELD_LOCATIONS, metric)
    return tuple(
        np.asarray(wald.physical_component(component, static_parameters))[..., 0]
        for component in B_center
    )


def line_seed_points(config, theta_axis, r_axis):
    """Seed asymptotic flux tubes and a dense ring near the event horizon."""

    (r_inner, r_outer), _, _ = wald.grid_bounds(config)
    absorber_length, _ = wald.absorber_width(config)
    analysis = config["analysis"]

    seed_radius = min(r_outer - 1.5 * absorber_length, r_axis[-2])
    cylindrical_radius = np.linspace(
        -float(analysis["field_line_radial_fraction"]) * seed_radius,
        float(analysis["field_line_radial_fraction"]) * seed_radius,
        int(analysis["field_line_count"]),
    )

    # Replace the central regular seeds with explicit impact parameters
    # inside the initial horizon cross-section. Seeding them at the outer
    # boundary selects the same incoming flux tubes in every frame without
    # pinning the later curves to the horizon, so their expulsion stays visible.
    threading_count = int(analysis["threading_field_line_count"])
    central_indices = np.argsort(np.abs(cylindrical_radius))[:threading_count]
    outer_radius = np.delete(cylindrical_radius, central_indices)
    outer_theta = np.arcsin(outer_radius / seed_radius)
    outer_theta = np.where(outer_theta < 0.0, outer_theta + 2.0 * np.pi, outer_theta)
    outer_theta = np.clip(outer_theta, theta_axis[0], theta_axis[-1])
    outer_seeds = np.column_stack(
        (outer_theta, np.full_like(outer_theta, seed_radius))
    )

    impact_radius = float(analysis["threading_impact_fraction"]) * r_inner
    threading_radius = np.linspace(-impact_radius, impact_radius, threading_count)
    threading_theta = np.arcsin(threading_radius / seed_radius)
    threading_theta = np.where(
        threading_theta < 0.0,
        threading_theta + 2.0 * np.pi,
        threading_theta,
    )
    threading_theta = np.clip(threading_theta, theta_axis[0], theta_axis[-1])
    threading_seeds = np.column_stack(
        (
            threading_theta,
            np.full_like(threading_theta, seed_radius),
        )
    )

    near_horizon_count = int(analysis["near_horizon_field_line_count"])
    dr = r_axis[1] - r_axis[0]
    near_horizon_radius = r_inner + (
        float(analysis["near_horizon_seed_offset_cells"]) * dr
    )
    near_horizon_theta = theta_axis[0] + (
        2.0
        * np.pi
        * (np.arange(near_horizon_count) + 0.25)
        / near_horizon_count
    )
    near_horizon_seeds = np.column_stack(
        (
            near_horizon_theta,
            np.full_like(near_horizon_theta, near_horizon_radius),
        )
    )

    return np.vstack((outer_seeds, threading_seeds, near_horizon_seeds))


def magnetic_line_segments(theta_axis, r_axis, B_center, seeds):
    """Trace magnetic lines in the native Kerr-Schild theta-r plane."""

    B_r, B_theta, _ = B_center
    temporary_figure, temporary_axis = plt.subplots()
    coordinate_segments = []

    # Trace each fixed impact parameter with an independent streamline mask.
    # A shared streamplot mask suppresses a seed whenever an earlier trajectory
    # enters the same cell.  Which seed wins can change between nearby frames,
    # making magnetic lines blink even though the saved field evolves smoothly.
    for seed in seeds:
        stream = temporary_axis.streamplot(
            theta_axis,
            r_axis,
            B_theta,
            B_r,
            start_points=seed[None, :],
            integration_direction="both",
            maxlength=8.0,
            minlength=0.05,
            broken_streamlines=False,
        )
        coordinate_segments.extend(
            np.array(segment, copy=True)
            for segment in stream.lines.get_segments()
        )
        stream.lines.remove()
    plt.close(temporary_figure)

    return coordinate_segments


def black_hole_curves(config, plot_r_outer):
    """Return horizon, ergosphere, and absorber curves in theta-r coordinates."""

    physics = config["physics"]
    (r_inner, r_outer), _, _ = wald.grid_bounds(config)
    absorber_length, _ = wald.absorber_width(config)
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    ergosphere_radius = float(physics["mass"]) + np.sqrt(
        float(physics["mass"]) ** 2
        - float(physics["spin"]) ** 2 * np.cos(angle) ** 2
    )

    curves = {
        "horizon": (
            angle,
            np.full_like(angle, r_inner),
        ),
        "ergosphere": (
            angle,
            ergosphere_radius,
        ),
        "absorber": (
            angle,
            np.full_like(angle, r_outer - absorber_length),
        ),
        "r_outer": plot_r_outer,
    }
    return curves


def draw_geometry(axis, curves):
    axis.fill(*curves["horizon"], color="black", zorder=5)
    axis.plot(*curves["horizon"], "k-", lw=1.2)
    axis.plot(*curves["ergosphere"], color="lightskyblue", lw=1.2)
    axis.set_theta_zero_location("S")
    axis.set_theta_direction(1)
    axis.set_thetagrids(np.arange(0, 360, 45))
    axis.set_ylim(0.0, curves["r_outer"])
    axis.set_yticks([])
    axis.grid(False)


def build_comparison_figure(
    output_path,
    times,
    numerical,
    analytical,
    theta_edges,
    r_edges,
    numerical_lines,
    analytical_lines,
    curves,
    plot_radial_stop,
    radial_stop,
    dpi,
):
    """Write the final numerical/analytical comparison and error history."""

    difference = numerical[-1] - analytical
    shared_limit = max(
        float(np.max(np.abs(numerical[-1]))),
        float(np.max(np.abs(analytical))),
        np.finfo(float).eps,
    )
    difference_limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    rms_error = np.sqrt(
        np.mean(
            (numerical[:, :radial_stop] - analytical[None, :radial_stop]) ** 2,
            axis=(1, 2),
        )
    )

    figure = plt.figure(figsize=(22, 5.3), constrained_layout=True)
    axes = [
        figure.add_subplot(1, 4, panel + 1, projection="polar")
        for panel in range(3)
    ]
    axes.append(figure.add_subplot(1, 4, 4))
    panels = (
        (numerical[-1], f"Numerical, t={times[-1]:.2f} M", numerical_lines),
        (analytical, "Analytical Kerr Wald", analytical_lines),
    )
    for axis, (field, title, lines) in zip(axes[:2], panels):
        image = axis.pcolormesh(
            theta_edges,
            r_edges,
            field[:plot_radial_stop],
            shading="flat",
            cmap="RdBu_r",
            vmin=-shared_limit,
            vmax=shared_limit,
        )
        axis.add_collection(LineCollection(lines, colors="k", linewidths=0.5))
        draw_geometry(axis, curves)
        axis.set_title(title)
    figure.colorbar(
        image,
        ax=axes[:2],
        label=r"$\gamma_{ij}D^iB^j/B_0^2$",
    )

    difference_image = axes[2].pcolormesh(
        theta_edges,
        r_edges,
        difference[:plot_radial_stop],
        shading="flat",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    draw_geometry(axes[2], curves)
    axes[2].set_title("Numerical - analytical")
    figure.colorbar(difference_image, ax=axes[2], label="absolute difference")

    axes[3].semilogy(times, np.maximum(rms_error, np.finfo(float).tiny))
    axes[3].set(
        xlabel="t / M",
        ylabel="RMS parallel-field error",
        title="Relaxation toward Kerr Wald",
    )
    axes[3].grid(alpha=0.3)

    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return rms_error


def build_movie(
    output_path,
    times,
    parallel_fields,
    centered_B,
    theta_edges,
    r_edges,
    theta_axis,
    r_axis,
    seeds,
    curves,
    plot_radial_stop,
    fps,
    dpi,
):
    """Write a fixed-scale H.264 movie with evolving magnetic lines."""

    color_limit = max(float(np.max(np.abs(parallel_fields))), np.finfo(float).eps)
    figure, axis = plt.subplots(
        figsize=(7, 7),
        constrained_layout=True,
        subplot_kw={"projection": "polar"},
    )
    image = axis.pcolormesh(
        theta_edges,
        r_edges,
        parallel_fields[0, :plot_radial_stop],
        shading="flat",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
    )
    initial_lines = magnetic_line_segments(
        theta_axis,
        r_axis,
        tuple(component[0, :plot_radial_stop] for component in centered_B),
        seeds,
    )
    line_collection = LineCollection(initial_lines, colors="k", linewidths=0.5)
    axis.add_collection(line_collection)
    draw_geometry(axis, curves)
    title = axis.set_title(f"t = {times[0]:.2f} M")
    figure.colorbar(
        image,
        ax=axis,
        label=r"$\gamma_{ij}D^iB^j/B_0^2$",
    )

    def animate(frame_index):
        image.set_array(
            parallel_fields[frame_index, :plot_radial_stop].ravel()
        )
        lines = magnetic_line_segments(
            theta_axis,
            r_axis,
            tuple(
                component[frame_index, :plot_radial_stop]
                for component in centered_B
            ),
            seeds,
        )
        line_collection.set_segments(lines)
        title.set_text(f"t = {times[frame_index]:.2f} M")
        return image, line_collection, title

    animation = FuncAnimation(
        figure,
        animate,
        frames=len(times),
        interval=1000.0 / fps,
        blit=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(figure)


def analyze(config):
    """Load the run, regenerate the Kerr target, and write both artifacts."""

    output = config["output"]
    analysis = config["analysis"]
    output_dir = Path(output["directory"])
    field_path = output_dir / (
        output["field_filename"] + output["field_extension"]
    )
    times, parallel_fields, magnetic_fields = read_field_series(field_path)
    parallel_fields = parallel_fields[..., 0]

    static_parameters, dynamic_parameters, _ = wald.build_pypic_parameters(config)
    metric = wald.initialize_metric(static_parameters, dynamic_parameters)
    D_target, B_target, _, _ = wald.initialize_kerr_target(
        config,
        static_parameters,
        dynamic_parameters,
        metric,
    )
    analytical = np.asarray(
        wald.physical_component(
            wald.parallel_electric_field(
                D_target,
                B_target,
                metric,
                config["physics"]["B0"],
            ),
            static_parameters,
        )
    )[..., 0]

    r_axis, theta_axis = (
        np.asarray(axis) for axis in wald.physical_center_axes(dynamic_parameters)
    )
    plot_r_outer = float(analysis["plot_r_outer"])
    plot_radial_stop = int(np.searchsorted(r_axis, plot_r_outer, side="right"))
    plot_r_axis = r_axis[:plot_radial_stop]
    r_edges = cell_edges(plot_r_axis)
    theta_edges = cell_edges(theta_axis)
    seeds = line_seed_points(config, theta_axis, plot_r_axis)
    curves = black_hole_curves(config, plot_r_outer)

    centered_frames = [
        center_saved_magnetic_field(
            tuple(field[frame] for field in magnetic_fields),
            static_parameters,
            metric,
        )
        for frame in range(len(times))
    ]
    centered_B = tuple(
        np.stack([frame[component] for frame in centered_frames])
        for component in range(3)
    )
    analytical_B = tuple(
        np.asarray(wald.physical_component(component, static_parameters))[..., 0]
        for component in wald.center_vector(B_target, B_FIELD_LOCATIONS, metric)
    )

    numerical_lines = magnetic_line_segments(
        theta_axis,
        plot_r_axis,
        tuple(component[-1, :plot_radial_stop] for component in centered_B),
        seeds,
    )
    analytical_lines = magnetic_line_segments(
        theta_axis,
        plot_r_axis,
        tuple(component[:plot_radial_stop] for component in analytical_B),
        seeds,
    )
    _, absorber_cells = wald.absorber_width(config)
    radial_stop = int(config["grid"]["nr"]) - absorber_cells

    final_B_tiled = restore_tiled_magnetic_field(
        tuple(component[-1] for component in magnetic_fields),
        static_parameters,
    )
    final_divergence = np.asarray(
        wald.physical_component(
            wald.weighted_magnetic_divergence(
                densitize_vector(final_B_tiled, metric.B),
                metric,
                dynamic_parameters,
            ),
            static_parameters,
        )
    )
    divergence_core = final_divergence[2:max(3, radial_stop - 2), 2:-2, :]
    divergence_rms = float(np.sqrt(np.mean(divergence_core**2)))

    comparison_path = output_dir / analysis["comparison_filename"]
    movie_path = output_dir / analysis["movie_filename"]
    rms_error = build_comparison_figure(
        comparison_path,
        times,
        parallel_fields,
        analytical,
        theta_edges,
        r_edges,
        numerical_lines,
        analytical_lines,
        curves,
        plot_radial_stop,
        radial_stop,
        int(analysis["dpi"]),
    )
    build_movie(
        movie_path,
        times,
        parallel_fields,
        centered_B,
        theta_edges,
        r_edges,
        theta_axis,
        plot_r_axis,
        seeds,
        curves,
        plot_radial_stop,
        int(analysis["fps"]),
        int(analysis["dpi"]),
    )

    print(f"Initial cropped RMS error: {rms_error[0]:.6e}")
    print(f"Final cropped RMS error:   {rms_error[-1]:.6e}")
    print(f"Final cropped RMS div(B):  {divergence_rms:.6e}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {movie_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("wald_demo.toml"),
        help="Wald demo TOML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(wald.load_configuration(args.config))


if __name__ == "__main__":
    main()
