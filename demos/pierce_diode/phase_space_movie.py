from pathlib import Path

import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation


C = 299792458.0
FPS = 30
NUM_FRAMES = 900

DEMO_DIR = Path(__file__).resolve().parent
PARTICLE_FILE = DEMO_DIR / "data" / "particles.h5"
MOVIE_FILE = DEMO_DIR / "vx_vs_x_phase_space.mp4"


def read_electron_phase_space(particle_data, iteration):
    """Read electron x and reconstruct the relativistic velocity vx."""

    electron_data = particle_data[f"data/{iteration}/particles/electron1"]

    x = electron_data["position/x"][...] + electron_data["positionOffset/x"][...]
    mass = electron_data["mass"][...]

    px = electron_data["momentum/x"][...]
    py = electron_data["momentum/y"][...]
    pz = electron_data["momentum/z"][...]

    # openPMD stores p = gamma m v.  All momentum components contribute to gamma.
    gamma = np.sqrt(1.0 + (px**2 + py**2 + pz**2) / (mass**2 * C**2))
    vx = px / (gamma * mass)

    iteration_data = particle_data[f"data/{iteration}"]
    time = iteration_data.attrs["time"] * iteration_data.attrs["timeUnitSI"]

    return x, vx, time


def create_phase_space_movie(
    particle_file=PARTICLE_FILE,
    movie_file=MOVIE_FILE,
    num_frames=NUM_FRAMES,
    fps=FPS,
):
    """Create an electron vx-versus-x phase-space movie from saved openPMD data."""

    particle_file = Path(particle_file)
    movie_file = Path(movie_file)

    with h5py.File(particle_file, "r") as particle_data:
        iterations = sorted(int(iteration) for iteration in particle_data["data"])
        frame_indices = np.linspace(
            0,
            len(iterations) - 1,
            min(num_frames, len(iterations)),
            dtype=int,
        )
        frame_iterations = [iterations[index] for index in np.unique(frame_indices)]

        fig, ax = plt.subplots(figsize=(9, 6))
        phase_space = ax.scatter([], [], s=1.0, alpha=0.45, color="tab:blue", rasterized=True)
        time_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-C / 1.0e6, C / 1.0e6)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$v_x$ ($10^6$ m/s)")
        ax.set_title("Pierce diode electron phase space")
        ax.grid(alpha=0.2)

        def update(frame_iteration):
            x, vx, time = read_electron_phase_space(particle_data, frame_iteration)
            phase_space.set_offsets(np.column_stack((x, vx / 1.0e6)))
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

    print(f"Saved {len(frame_iterations)} frames to {movie_file}")


if __name__ == "__main__":
    create_phase_space_movie()
