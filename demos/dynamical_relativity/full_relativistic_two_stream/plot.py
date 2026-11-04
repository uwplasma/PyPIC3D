"""Plot native constraint histories and local-velocity phase-space movies.

Run python plot.py beside initial_data.py, or select an older run with
--run-dir runs/<timestamp>. Inputs are read-only; figures go into run/plots/.
The vertical movie axis is velocity along the local unit coordinate-x vector,
v_hat_x = u_x / (Gamma sqrt(gamma_xx)), in the demo's c=1 units.
"""

import argparse
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
import openpmd_api as io
import toml
from tqdm import tqdm

from PyPIC3D.relativity.interpolate_metric import interpolate_hermite

jax.config.update("jax_enable_x64", True)

CONSTRAINTS = (
    ("hamiltonian", "Hamiltonian"), ("momentum", "Momentum"),
    ("det_gamma", "Conformal determinant"), ("trace_A", "Trace of A"),
    ("gamma", "Conformal connection"),
)
ELECTRONS = ("electron_plus", "electron_minus")
DISPLAY_FLOOR = 1.e-18


def read_constraint(path):
    """Read the native comma-separated (time, norm) rows."""
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            time, value = map(float, line.split(","))
        except ValueError as error:
            raise ValueError(f"Malformed constraint row {path}:{line_number}") from error
        rows.append((time, value))
    if not rows:
        raise ValueError(f"Empty constraint history: {path}")
    values = jnp.asarray(rows)
    if (not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values[:, 1] < 0))
            or bool(jnp.any(jnp.diff(values[:, 0]) <= 0))):
        raise ValueError(f"Expected finite, nonnegative norms at increasing times: {path}")
    return values[:, 0], values[:, 1]


def plot_constraints(data_dir, output_dir):
    histories = {name: [read_constraint(data_dir / f"{name}_{norm}.txt")
                        for norm in ("l2", "linf")] for name, _ in CONSTRAINTS}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    try:
        for axis, (name, title) in zip(axes.flat, CONSTRAINTS):
            for (time, value), label, style in zip(histories[name], ("L2", "L∞"), ("-", "--")):
                axis.semilogy(time, jnp.maximum(value, DISPLAY_FLOOR), style, label=label)
            axis.set(title=title, xlabel="Simulation time", ylabel="Raw constraint norm")
            axis.grid(alpha=.25)
            axis.legend()
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.05, .5, f"Display floor: {DISPLAY_FLOOR:g}\nSaved values are unchanged.",
                           transform=axes.flat[-1].transAxes)
        fig.savefig(output_dir / "constraints.png", dpi=160)
    finally:
        plt.close(fig)


def matching_iterations(fields, particles, stride):
    indices = list(particles.iterations)
    if not indices:
        raise ValueError("No particle snapshots found")
    selected = indices[::stride]
    if selected[-1] != indices[-1]:
        selected.append(indices[-1])
    for index in selected:
        if index not in fields.iterations:
            raise ValueError(f"Particle snapshot {index} has no matching metric snapshot")
        field = fields.iterations[index]
        particle = particles.iterations[index]
        tf = field.time * field.time_unit_SI
        tp = particle.time * particle.time_unit_SI
        if not math.isfinite(tf) or not math.isfinite(tp) or abs(tf - tp) > 1.e-12 * max(1., abs(tf), abs(tp)):
            raise ValueError(f"Snapshot {index} has different metric and particle times: {tf}, {tp}")
    return selected


@jax.jit
def local_x_velocity(position, momentum_x, mass, lorentz_factor, gamma_xx, origin, spacing):
    """Sample the physical metric with the production periodic Hermite stencil."""
    nx = gamma_xx.shape[0]
    x = origin[0] + jnp.mod(position[:, 0] - origin[0], nx * spacing[0])
    position = position.at[:, 0].set(x)
    padded = jnp.pad(gamma_xx, ((2, 2), (0, 0), (0, 0)), mode="wrap")
    grid = (origin[0] + spacing[0] * jnp.arange(-2, nx + 2),
            origin[1:2], origin[2:3])
    sampled = interpolate_hermite(padded, position, grid, (True, False, False), (0, 0, 0))
    velocity = momentum_x / mass / lorentz_factor / jnp.sqrt(sampled)
    return x, velocity


def read_frame(fields, particles, index, species_names):
    field = fields.iterations[index]
    particle = particles.iterations[index]
    scalar = io.Mesh_Record_Component.SCALAR
    names = ("conformal_factor", "conformal_metric_xx", "conformal_metric_xy",
             "conformal_metric_xz", "conformal_metric_yy", "conformal_metric_yz",
             "conformal_metric_zz")
    missing = set(names) - set(field.meshes)
    if missing:
        raise ValueError(f"Snapshot {index} lacks metric meshes: {sorted(missing)}")
    mesh = field.meshes["conformal_factor"]
    shape = tuple(mesh[scalar].shape)
    if len(shape) != 3 or shape[0] < 4 or shape[1:] != (1, 1):
        raise ValueError(f"Expected a 1D metric (Nx, 1, 1), got {shape}")
    spacing = jnp.asarray(mesh.grid_spacing) * mesh.grid_unit_SI
    position = jnp.asarray(mesh[scalar].position)
    origin = jnp.asarray(mesh.grid_global_offset) * mesh.grid_unit_SI + position * spacing
    if not bool(jnp.all(jnp.isfinite(origin) & jnp.isfinite(spacing) & (spacing > 0))):
        raise ValueError(f"Invalid metric coordinates at snapshot {index}")
    pending_metric = {}
    for name in names:
        component = field.meshes[name][scalar]
        other = field.meshes[name]
        if (tuple(component.shape) != shape or list(other.axis_labels) != ["x", "y", "z"]
                or list(other.grid_spacing) != list(mesh.grid_spacing)
                or list(other.grid_global_offset) != list(mesh.grid_global_offset)
                or list(component.position) != list(mesh[scalar].position)):
            raise ValueError(f"Metric mesh {name} has inconsistent coordinates at snapshot {index}")
        pending_metric[name] = component.load_chunk()
    fields.flush()
    W, xx, xy, xz, yy, yz, zz = [jnp.asarray(pending_metric[name]) for name in names]
    determinant = xx * (yy * zz - yz**2) - xy * (xy * zz - xz * yz) + xz * (xy * yz - xz * yy)
    if (not bool(jnp.all(jnp.isfinite(jnp.stack((W, xx, xy, xz, yy, yz, zz)))))
            or not bool(jnp.all((W > 0) & (xx > 0) & (xx * yy - xy**2 > 0) & (determinant > 0)))):
        raise ValueError(f"Invalid spatial metric at snapshot {index}")
    # Match build_bssn_metric's determinant normalization before interpolation.
    gamma_xx = xx * determinant**(-1. / 3.) / W**2

    phase_space = {}
    for name in species_names:
        if name not in particle.particles:
            raise ValueError(f"Missing species {name} at snapshot {index}")
        species = particle.particles[name]
        records = [("position", axis) for axis in "xyz"] + [("positionOffset", axis) for axis in "xyz"]
        records += [("momentum", "x"), ("mass", scalar), ("gamma", scalar)]
        pending = []
        for record_name, component_name in records:
            record = species[record_name]
            if "timeOffset" in record.attributes and float(record.get_attribute("timeOffset")) != 0:
                raise ValueError(f"{name}/{record_name} is not synchronized at snapshot {index}")
            component = record[component_name]
            pending.append((component.load_chunk(), component.unit_SI))
        particles.flush()
        values = [jnp.asarray(array) * unit for array, unit in pending]
        positions = jnp.stack(values[:3], axis=-1) + jnp.stack(values[3:6], axis=-1)
        momentum, mass, lorentz = values[6:]
        if (not all(bool(jnp.all(jnp.isfinite(value))) for value in values)
                or not bool(jnp.all((mass > 0) & (lorentz >= 1. - 1.e-12)))):
            raise ValueError(f"Invalid particle data for {name} at snapshot {index}")
        x, velocity = local_x_velocity(positions, momentum, mass, lorentz, gamma_xx, origin, spacing)
        if not bool(jnp.all(jnp.isfinite(velocity) & (jnp.abs(velocity) <= 1. + 1.e-9))):
            raise ValueError(f"Invalid local velocity for {name} at snapshot {index}")
        phase_space[name] = (x, velocity)
    limits = (float(origin[0]), float(origin[0] + shape[0] * spacing[0]))
    return particle.time * particle.time_unit_SI, limits, phase_space


def write_movie(fields, particles, indices, species_names, output, velocity_limit, fps, dpi):
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    try:
        with writer.saving(fig, str(output), dpi=dpi):
            artists = {name: axis.scatter([], [], s=2, alpha=.55, label=name)
                       for name in species_names}
            axis.set(xlabel="Coordinate x", ylabel=r"Local $v^{\hat{x}}/c$",
                     ylim=(-velocity_limit, velocity_limit))
            axis.legend(loc="upper right")
            axis.grid(alpha=.2)
            spatial_limits = None
            for index in tqdm(indices, desc=output.name, unit="frame"):
                time, limits, phase_space = read_frame(fields, particles, index, species_names)
                if spatial_limits is None:
                    spatial_limits = limits
                    axis.set_xlim(*limits)
                elif limits != spatial_limits:
                    raise ValueError("The coordinate domain changed between snapshots")
                for name, (x, velocity) in phase_space.items():
                    artists[name].set_offsets(jnp.stack((x, velocity), axis=-1))
                axis.set_title(f"Phase space at t = {time:.6g}")
                writer.grab_frame()
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if min(args.stride, args.fps, args.dpi) < 1:
        parser.error("--stride, --fps and --dpi must be positive")
    if args.run_dir is None:
        config = toml.load(Path(__file__).with_name("two_stream.toml"))
        parameters = {**config.get("simulation_parameters", {}), **config.get("static_parameters", {})}
        args.run_dir = Path(parameters["output_dir"])
    data_dir = args.run_dir / "data"
    output_dir = args.run_dir / "plots"
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required for MP4 output; make ffmpeg available on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_constraints(data_dir, output_dir)
    fields = particles = None
    try:
        fields = io.Series(str(data_dir / "fields.h5"), io.Access.read_only)
        particles = io.Series(str(data_dir / "particles.h5"), io.Access.read_only)
        indices = matching_iterations(fields, particles, args.stride)
        ion_limit = 0.
        for index in tqdm(indices, desc="Ion velocity range", unit="frame"):
            _, _, phase_space = read_frame(fields, particles, index, ("ion",))
            velocity = phase_space["ion"][1]
            ion_limit = max(ion_limit, float(jnp.max(jnp.abs(velocity), initial=0.)))
        ion_limit = max(1.05 * ion_limit, 1.e-6)
        write_movie(fields, particles, indices, ELECTRONS,
                    output_dir / "electrons_vx_x.mp4", 1., args.fps, args.dpi)
        write_movie(fields, particles, indices, ("ion",),
                    output_dir / "ions_vx_x.mp4", ion_limit, args.fps, args.dpi)
    finally:
        for series in (particles, fields):
            if series is not None:
                series.close()
    print(f"Saved plots and movies in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
