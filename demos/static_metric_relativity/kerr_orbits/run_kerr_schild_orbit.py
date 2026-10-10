#!/usr/bin/env python3
"""
Integrate and plot a neutral particle orbit in spherical Kerr-Schild coordinates.

The runtime particle state stores ``(r, theta, phi)`` and covariant spatial
momentum ``u_i``. The numerical advance repeatedly calls PyPIC3D's production
hybrid geodesic particle pusher against one fixed Kerr-Schild metric and zero
electromagnetic fields.

The pusher is a staggered leapfrog, so the stored ``u_i`` sits half a step
behind the stored position. The configured initial momentum is offset onto that
half step before the run starts. The reported ``E = -u_0`` and ``u_phi``
histories still combine ``x^n`` with ``u^{n+1/2}``, which leaves an ``O(dt)``
offset in the diagnostic itself; that is separate from the integration error and
is not corrected here.
"""

from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


WORKSPACE = Path(__file__).resolve().parent
PYPIC3D_SOURCE = WORKSPACE.parents[2]
if not (PYPIC3D_SOURCE / "PyPIC3D").is_dir():
    raise FileNotFoundError(f"Local PyPIC3D source tree not found: {PYPIC3D_SOURCE}")
sys.path.insert(0, str(PYPIC3D_SOURCE))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kerr-schild-orbit")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import toml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.metrics.kerr_schild import (
    _kerr_schild_spherical_metric_at_position,
)


CONFIG_FILE = WORKSPACE / "kerr_schild.toml"
TRAJECTORY_INTERVAL = 100
OUTPUT_FILE = "kerr_orbits.png"
ENERGY_ERROR_FILE = "energy_error.txt"


def install_exact_initial_state(particles, particle_config):
    """
    Replace the cell-jittered initializer state with the requested orbit state.
    """

    x_0 = jnp.asarray(
        (
            particle_config["initial_x"],
            particle_config["initial_y"],
            particle_config["initial_z"],
        ),
        dtype=particles.x.dtype,
    )
    u_0 = jnp.asarray(
        (
            particle_config["initial_vx"],
            particle_config["initial_vy"],
            particle_config["initial_vz"],
        ),
        dtype=particles.u.dtype,
    )
    active = particles.active[..., jnp.newaxis]

    x = jnp.where(active, x_0, particles.x)
    u = jnp.where(active, u_0, particles.u)
    return particles._replace(x=x, u=u)


def make_chunk_advance(
    static_parameters,
    dynamic_parameters,
    species_config,
    D,
    B,
    metric,
    chunk_steps,
):
    """
    Advance a neutral particle by repeatedly calling the production pusher.

    The Kerr-Schild metric and zero electromagnetic fields are fixed throughout
    the scan. No field solve, current deposition, particle communication, or
    boundary update is performed.
    """

    @jax.jit
    def advance_chunk(particles):
        def advance_one_step(particles, _):
            particles, _ = hybrid_boris_geodesic_push(
                particles,
                species_config,
                D,
                B,
                metric,
                static_parameters,
                dynamic_parameters,
            )
            return particles, particles

        return jax.lax.scan(
            advance_one_step,
            particles,
            xs=None,
            length=chunk_steps,
        )

    return advance_chunk


def active_particle_position(particles):
    positions = np.asarray(jax.device_get(particles.x[particles.active]))
    return positions[0]


def active_particle_momentum(particles):
    momenta = np.asarray(jax.device_get(particles.u[particles.active]))
    return momenta[0]


def compute_u0_energy(trajectory, momentum, mass, spin):
    """
    Compute the stationary-particle energy ``E = -u_0``.

    The four-velocity normalization gives
    ``Gamma = sqrt(1 + gamma^ij u_i u_j)`` and hence
    ``u_0 = beta^i u_i - alpha Gamma`` for stored covariant ``u_i``.
    """

    metric_at_position = partial(
        _kerr_schild_spherical_metric_at_position,
        mass=mass,
        spin=spin,
    )
    lapse, shift, _gamma, gamma_inv, _sqrt_gamma = jax.vmap(
        metric_at_position
    )(jnp.asarray(trajectory))
    momentum = jnp.asarray(momentum)
    Gamma = jnp.sqrt(
        1.0 + jnp.einsum("...i,...ij,...j->...", momentum, gamma_inv, momentum)
    )
    u_0 = jnp.einsum("...i,...i->...", shift, momentum) - lapse * Gamma
    return np.asarray(jax.device_get(-u_0))


def write_energy_error(times, energy, output_file):
    """Write an xmgrace-readable two-column relative energy error history."""

    initial_energy = energy[0]
    relative_error = np.abs((energy - initial_energy) / initial_energy)
    header = (
        "Kerr-Schild orbit energy error\n"
        "column 1: time / M\n"
        "column 2: abs((E - E_initial) / E_initial)\n"
        f"E_initial = {initial_energy:.16e}, E = -u_0"
    )
    np.savetxt(
        output_file,
        np.column_stack((times, relative_error)),
        fmt="%.16e",
        header=header,
    )
    return relative_error


def plot_orbit(trajectory, mass, spin, output_file):
    """Convert the spherical trajectory to Cartesian coordinates and plot it."""

    radius = trajectory[:, 0]
    theta = trajectory[:, 1]
    phi = trajectory[:, 2]

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    horizon_radius = mass + math.sqrt(mass**2 - spin**2)

    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    ax.plot(x, y, color="tab:blue", linewidth=0.8)
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            horizon_radius,
            color="black",
            label="Event horizon",
            zorder=3,
        )
    )
    ax.scatter(x[0], y[0], color="tab:red", s=20, label="Initial position", zorder=4)
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title("Neutral particle orbit in Kerr-Schild coordinates")
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def run_orbit(config):
    particle_config = config["particle1"]

    # PyPIC3D initialization currently writes initial histograms unconditionally.
    # Keep those initialization-only files outside the demo and remove them when
    # the initialized particle, field, and metric states have been constructed.
    with TemporaryDirectory(prefix="pypic3d-kerr-orbit-") as initialization_dir:
        config["simulation_parameters"]["output_dir"] = initialization_dir
        (
            _loop,
            particles,
            fields,
            static_parameters,
            dynamic_parameters,
            _plotting_parameters,
            _plasma_parameters,
            species_config,
        ) = initialize_simulation(config)

    particles = install_exact_initial_state(particles, particle_config)
    D, B = fields[:2]  # native densities, as the production pusher reads them
    metric = fields[6]

    particles = seed_leapfrog_velocity(
        particles,
        species_config,
        D,
        B,
        static_parameters,
        dynamic_parameters,
        metric=metric,
    )
    # install_exact_initial_state overwrites the velocity that
    # initialize_simulation already offset, so put it back on the half step

    initial_position = active_particle_position(particles)
    expected_position = np.asarray(
        (
            particle_config["initial_x"],
            particle_config["initial_y"],
            particle_config["initial_z"],
        )
    )
    if not np.array_equal(initial_position, expected_position):
        raise RuntimeError(
            f"Exact initial position was not installed: {initial_position}"
        )

    angular_momentum = float(particle_config["initial_vz"])
    initial_active = np.asarray(jax.device_get(particles.active))
    initial_u_phi = np.asarray(jax.device_get(particles.u[..., 2]))
    max_angular_momentum_error = float(
        np.max(np.abs(initial_u_phi[initial_active] - angular_momentum))
    )

    trajectory = [initial_position]
    momentum = [active_particle_momentum(particles)]
    sample_steps = [0]
    n_steps = int(static_parameters.Nt)
    completed_steps = 0
    n_chunks = int(math.ceil(n_steps / TRAJECTORY_INTERVAL))
    advance_chunk = make_chunk_advance(
        static_parameters,
        dynamic_parameters,
        species_config,
        D,
        B,
        metric,
        TRAJECTORY_INTERVAL,
    )

    for _ in tqdm(range(n_chunks), desc="Kerr-Schild orbit", unit="sample"):
        chunk_steps = min(TRAJECTORY_INTERVAL, n_steps - completed_steps)
        if chunk_steps != TRAJECTORY_INTERVAL:
            advance_chunk = make_chunk_advance(
                static_parameters,
                dynamic_parameters,
                species_config,
                D,
                B,
                metric,
                chunk_steps,
            )

        particles, particle_history = advance_chunk(particles)
        jax.block_until_ready(particles)

        history_active = np.asarray(jax.device_get(particle_history.active))
        history_u_phi = np.asarray(jax.device_get(particle_history.u[..., 2]))
        if np.any(history_active):
            max_angular_momentum_error = max(
                max_angular_momentum_error,
                float(
                    np.max(
                        np.abs(
                            history_u_phi[history_active]
                            - angular_momentum
                        )
                    )
                ),
            )

        trajectory.append(active_particle_position(particles))
        completed_steps += chunk_steps
        momentum.append(active_particle_momentum(particles))
        sample_steps.append(completed_steps)

    trajectory = np.asarray(trajectory)
    momentum = np.asarray(momentum)
    sample_times = np.asarray(sample_steps) * float(dynamic_parameters.dt)
    final_active = int(np.asarray(jax.device_get(jnp.sum(particles.active))))

    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError("The orbit contains non-finite particle coordinates")
    if not np.all(np.isfinite(momentum)):
        raise RuntimeError("The orbit contains non-finite particle momenta")
    if max_angular_momentum_error > 1.0e-12:
        raise RuntimeError(
            "Azimuthal Killing momentum drifted during the orbit: "
            f"{max_angular_momentum_error}"
        )
    if final_active != 1:
        raise RuntimeError(f"Expected one active particle, found {final_active}")

    output_file = Path.cwd() / OUTPUT_FILE
    energy_error_file = Path.cwd() / ENERGY_ERROR_FILE
    mass = float(static_parameters.metric_mass)
    spin = float(static_parameters.metric_spin)
    plot_orbit(
        trajectory,
        mass=mass,
        spin=spin,
        output_file=output_file,
    )
    energy = compute_u0_energy(trajectory, momentum, mass=mass, spin=spin)
    if not np.all(np.isfinite(energy)):
        raise RuntimeError("The orbit contains non-finite normalized energies")
    relative_energy_error = write_energy_error(
        sample_times,
        energy,
        energy_error_file,
    )

    print()
    print(f"completed time: {completed_steps * float(dynamic_parameters.dt):.12g} M")
    print(f"trajectory samples: {trajectory.shape[0]}")
    print(f"active particles: {final_active}")
    print(f"max |u_phi - L|: {max_angular_momentum_error:.3e}")
    print(f"initial -u_0 energy: {energy[0]:.16e}")
    print(f"max relative energy error: {np.max(relative_energy_error):.3e}")
    print(f"orbit plot: {output_file}")
    print(f"energy error: {energy_error_file}")

    return output_file, energy_error_file


def main():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    config = toml.load(CONFIG_FILE)
    run_orbit(config)


if __name__ == "__main__":
    main()
