#!/usr/bin/env python3
"""Create Pierce-diode electron phase-space and potential movies."""

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation


C = 299_792_458.0
FPS = 30
NUM_FRAMES = 900
VELOCITY_SCALE = 1.0e6

DEMO_DIR = Path(__file__).resolve().parent
PARTICLE_FILE = DEMO_DIR / "data" / "particles.h5"
FIELD_FILE = DEMO_DIR / "data" / "fields.h5"
PHASE_SPACE_MOVIE = DEMO_DIR / "vx_vs_x_phase_space.mp4"
POTENTIAL_MOVIE = DEMO_DIR / "phi_vs_x.mp4"


def _iteration_time(iteration_data):
    return float(iteration_data.attrs["time"] * iteration_data.attrs["timeUnitSI"])


def read_electron_phase_space(particle_data, iteration):
    """Read electron x and reconstruct the physical relativistic velocity vx."""

    iteration_data = particle_data[f"data/{iteration}"]
    electron_data = iteration_data["particles/electron1"]

    x = electron_data["position/x"][...] + electron_data["positionOffset/x"][...]
    mass = electron_data["mass"][...]
    px = electron_data["momentum/x"][...]
    py = electron_data["momentum/y"][...]
    pz = electron_data["momentum/z"][...]

    # openPMD stores p = gamma*m*v, so all momentum components enter gamma.
    gamma = np.sqrt(1.0 + (px**2 + py**2 + pz**2) / (mass**2 * C**2))
    vx = px / (gamma * mass)

    return x, vx, _iteration_time(iteration_data)


def read_potential_profile(field_data, iteration):
    """Read the native openPMD electrostatic potential along x."""

    iteration_data = field_data[f"data/{iteration}"]
    meshes = iteration_data["meshes"]
    if "phi" not in meshes:
        raise KeyError(
            f"Iteration {iteration} does not contain meshes/phi. "
            "Run the electrostatic simulation with native phi output enabled."
        )

    phi_dataset = meshes["phi"]
    phi = np.asarray(phi_dataset[...]).squeeze()
    if phi.ndim != 1:
        raise ValueError(
            "Pierce potential output must reduce to one x profile; "
            f"got phi shape {phi_dataset.shape}."
        )

    grid_spacing = np.asarray(phi_dataset.attrs["gridSpacing"], dtype=float)
    grid_offset = np.asarray(phi_dataset.attrs["gridGlobalOffset"], dtype=float)
    grid_unit = float(phi_dataset.attrs.get("gridUnitSI", 1.0))
    position = np.asarray(phi_dataset.attrs.get("position", (0.0,)), dtype=float)

    dx = grid_spacing[0] * grid_unit
    x_offset = grid_offset[0] * grid_unit
    x_position = position[0] if position.size else 0.0
    x = x_offset + (np.arange(phi.size) + x_position) * dx
    phi = phi * float(phi_dataset.attrs.get("unitSI", 1.0))

    return x, phi, _iteration_time(iteration_data)


def common_frame_iterations(particle_data, field_data, num_frames):
    """Return uniformly sampled iterations present in both openPMD files."""

    particle_iterations = {int(iteration) for iteration in particle_data["data"]}
    field_iterations = {int(iteration) for iteration in field_data["data"]}
    iterations = sorted(particle_iterations & field_iterations)
    if not iterations:
        raise ValueError("Particle and field files have no common iterations.")

    frame_indices = np.linspace(
        0,
        len(iterations) - 1,
        min(int(num_frames), len(iterations)),
        dtype=int,
    )
    return [iterations[index] for index in np.unique(frame_indices)]


def _padded_limits(lower, upper, padding_fraction=0.05):
    span = upper - lower
    if span == 0.0:
        span = max(abs(lower), 1.0)
    padding = padding_fraction * span
    return lower - padding, upper + padding


def scan_movie_limits(particle_data, field_data, frame_iterations):
    """Stream selected frames once to obtain fixed axes for both movies."""

    x_min = np.inf
    x_max = -np.inf
    vx_min = np.inf
    vx_max = -np.inf
    phi_min = np.inf
    phi_max = -np.inf

    for iteration in frame_iterations:
        particle_x, vx, _ = read_electron_phase_space(particle_data, iteration)
        field_x, phi, _ = read_potential_profile(field_data, iteration)

        x_min = min(x_min, float(np.min(particle_x)), float(np.min(field_x)))
        x_max = max(x_max, float(np.max(particle_x)), float(np.max(field_x)))
        vx_min = min(vx_min, float(np.min(vx)))
        vx_max = max(vx_max, float(np.max(vx)))
        phi_min = min(phi_min, float(np.min(phi)))
        phi_max = max(phi_max, float(np.max(phi)))

    return (
        _padded_limits(x_min, x_max, padding_fraction=0.01),
        _padded_limits(vx_min / VELOCITY_SCALE, vx_max / VELOCITY_SCALE),
        _padded_limits(phi_min, phi_max),
    )


def create_phase_space_movie(
    particle_data,
    frame_iterations,
    x_limits,
    velocity_limits,
    movie_file,
    fps,
):
    """Write the electron vx-versus-x phase-space MP4."""

    fig, ax = plt.subplots(figsize=(9, 6))
    phase_space = ax.scatter(
        [],
        [],
        s=1.0,
        alpha=0.45,
        color="tab:blue",
        rasterized=True,
    )
    time_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    ax.set_xlim(*x_limits)
    ax.set_ylim(*velocity_limits)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$v_x$ ($10^6$ m/s)")
    ax.set_title("Pierce diode electron phase space")
    ax.grid(alpha=0.2)

    def update(iteration):
        x, vx, time = read_electron_phase_space(particle_data, iteration)
        phase_space.set_offsets(np.column_stack((x, vx / VELOCITY_SCALE)))
        time_label.set_text(f"t = {time * 1.0e9:.3f} ns")
        return phase_space, time_label

    movie = FuncAnimation(
        fig,
        update,
        frames=frame_iterations,
        interval=1000.0 / fps,
        blit=True,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"title": "Pierce diode electron phase space"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(movie_file, writer=writer, dpi=150)
    plt.close(fig)


def create_potential_movie(
    field_data,
    frame_iterations,
    x_limits,
    potential_limits,
    movie_file,
    fps,
):
    """Write the native phi-versus-x electrostatic-potential MP4."""

    fig, ax = plt.subplots(figsize=(9, 6))
    potential_line, = ax.plot([], [], color="tab:red", linewidth=1.5)
    time_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    ax.set_xlim(*x_limits)
    ax.set_ylim(*potential_limits)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$\phi$ (V)")
    ax.set_title("Pierce diode electrostatic potential")
    ax.grid(alpha=0.2)

    def update(iteration):
        x, phi, time = read_potential_profile(field_data, iteration)
        potential_line.set_data(x, phi)
        time_label.set_text(f"t = {time * 1.0e9:.3f} ns")
        return potential_line, time_label

    movie = FuncAnimation(
        fig,
        update,
        frames=frame_iterations,
        interval=1000.0 / fps,
        blit=True,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"title": "Pierce diode electrostatic potential"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(movie_file, writer=writer, dpi=150)
    plt.close(fig)


def create_movies(
    particle_file=PARTICLE_FILE,
    field_file=FIELD_FILE,
    phase_space_movie=PHASE_SPACE_MOVIE,
    potential_movie=POTENTIAL_MOVIE,
    num_frames=NUM_FRAMES,
    fps=FPS,
):
    """Create both Pierce-diode movies from common openPMD iterations."""

    particle_file = Path(particle_file)
    field_file = Path(field_file)
    phase_space_movie = Path(phase_space_movie)
    potential_movie = Path(potential_movie)
    phase_space_movie.parent.mkdir(parents=True, exist_ok=True)
    potential_movie.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(particle_file, "r") as particle_data,
        h5py.File(field_file, "r") as field_data,
    ):
        frame_iterations = common_frame_iterations(
            particle_data,
            field_data,
            num_frames,
        )
        # Read phi before creating either movie so legacy field files fail
        # immediately without overwriting an existing phase-space movie.
        read_potential_profile(field_data, frame_iterations[0])
        x_limits, velocity_limits, potential_limits = scan_movie_limits(
            particle_data,
            field_data,
            frame_iterations,
        )

        create_phase_space_movie(
            particle_data,
            frame_iterations,
            x_limits,
            velocity_limits,
            phase_space_movie,
            fps,
        )
        create_potential_movie(
            field_data,
            frame_iterations,
            x_limits,
            potential_limits,
            potential_movie,
            fps,
        )

    print(f"Saved {len(frame_iterations)} frames to {phase_space_movie}")
    print(f"Saved {len(frame_iterations)} frames to {potential_movie}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=PARTICLE_FILE)
    parser.add_argument("--fields", type=Path, default=FIELD_FILE)
    parser.add_argument("--phase-space-movie", type=Path, default=PHASE_SPACE_MOVIE)
    parser.add_argument("--potential-movie", type=Path, default=POTENTIAL_MOVIE)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    create_movies(
        particle_file=args.particles,
        field_file=args.fields,
        phase_space_movie=args.phase_space_movie,
        potential_movie=args.potential_movie,
        num_frames=args.num_frames,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
