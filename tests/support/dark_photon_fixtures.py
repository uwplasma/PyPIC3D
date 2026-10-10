"""Proca wave, constraint, convergence, and public-runtime regression tests.

Run with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=16.
"""

import contextlib
import io
import math
import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml
import openpmd_api as pmd

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells, update_tiled_vector_ghost_cells
from PyPIC3D.initialization import initialize_simulation, initialize_fields, _dark_timestep, default_parameters
from PyPIC3D.diagnostics.diagnostic_quantities import compute_dark_energy
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, field_map_for_output, scalar_field_for_output
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    compute_dark_B, dark_divergence, dark_gradient, initialize_dark_photon_fields,
    synchronized_dark_fields, update_dark_A, update_dark_E, update_dark_phi,
)
from PyPIC3D.solvers.dark_matter_yee.time_loop import time_loop_dark_photon, dark_photon_push_fields
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.utilities.field_helpers import filter_electric_field_for_particles
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from PyPIC3D.utilities.parameters import build_static_parameters, static_parameters_for_output
from PyPIC3D.utilities.toml_helpers import load_dark_fields_from_toml
from tests.kernel_fixtures import kernel_parameters, build_tiled_particles, particle_species


def parameters(n=32, dt=1/128, tiles=1, **kwargs):
    return kernel_parameters(
        Nx=n, Ny=1, Nz=1, x_wind=2*math.pi, y_wind=1., z_wind=1.,
        tile_shape=(n//tiles, 1, 1), dt=dt, solver="dark_matter_yee",
        C=kwargs.pop("C", 1.3), dark_mu=kwargs.pop("dark_mu", 0.7), **kwargs,
    )


def interior(s):
    return (slice(None),)*3 + (slice(s.guard_cells, -s.guard_cells),)*3


def wave(s, d, longitudinal, t=0., discrete=False):
    """A real traveling mode at its actual Yee component coordinates."""
    E, A, _J, phi, _rho = initialize_fields(s, d)
    k = 1.
    kh = 2 * np.sin(k*float(d.dx)/2)/float(d.dx) if discrete else k
    omega = float(d.C) * np.sqrt(kh**2+s.dark_mu**2)
    component = 0 if longitudinal else 1
    x = (d.grids.tiled_vertex_grid if longitudinal else d.grids.tiled_center_grid)[0]
    phase = x[..., None, None] - omega*t
    a = jnp.broadcast_to(jnp.cos(phase), phi.shape)
    amplitude = float(d.C)**2*s.dark_mu**2/omega if longitudinal else omega
    e = jnp.broadcast_to(-amplitude*jnp.sin(phase), phi.shape)
    E, A = list(E), list(A)
    E[component], A[component] = e, a
    if longitudinal:
        x_phi = d.grids.tiled_center_grid[0][..., None, None]
        phi = jnp.broadcast_to(float(d.C)**2*kh/omega*jnp.cos(x_phi-omega*t), phi.shape)
    return tuple(E), tuple(A), phi


def evolve(state, s, d, steps):
    """Jitted source-free Proca steps in the same order as time_loop_dark_photon."""
    J = tuple(jnp.zeros_like(v) for v in state[0])

    def step(_, state):
        E, A, phi = state
        A = update_dark_A(E, A, phi, s, d, d.dt)
        B = compute_dark_B(A, s, d)
        phi = update_dark_phi(A, phi, s, d, d.dt)
        E = update_dark_E(E, B, A, J, s, d, d.dt)
        return E, A, phi

    return jax.jit(lambda state: jax.lax.fori_loop(0, steps, step, state))(state)


def wave_errors(s, d, result, longitudinal, discrete=False):
    E, A, phi = result
    exact_E, _, exact_phi = wave(s, d, longitudinal, 1., discrete)
    _, exact_A, _ = wave(s, d, longitudinal, 1.-float(d.dt)/2, discrete)
    component = 0 if longitudinal else 1
    pairs = [(E[component], exact_E[component]), (A[component], exact_A[component])]
    if longitudinal:
        pairs.append((phi, exact_phi))
    else:
        kh = 2*np.sin(float(d.dx)/2)/float(d.dx) if discrete else 1.
        omega = float(d.C)*np.sqrt(kh**2+s.dark_mu**2)
        x_b = d.grids.tiled_vertex_grid[0][..., None, None]
        exact_B = jnp.broadcast_to(-kh*jnp.sin(x_b-omega*(1.-float(d.dt)/2)), phi.shape)
        pairs.append((compute_dark_B(A, s, d)[2], exact_B))
    return np.array([float(jnp.sqrt(jnp.mean((a[interior(s)]-b[interior(s)])**2))) for a, b in pairs])


class DarkPhotonFixtures:
    def assert_tree_close(self, a, b, atol=2e-12):
        self.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=atol)


    def config(self, output_dir, **updates):
        simulation = dict(solver="dark_matter_yee", output_dir=output_dir, Nx=8, Ny=1, Nz=1,
                          x_wind=2*math.pi, y_wind=1., z_wind=1., dt=.01, Nt=2,
                          C=1.3, eps=2., mu=1/(2*1.3**2), dark_mu=.7, sin_chi=.2, filter_j="none")
        simulation.update(updates)
        return {"simulation_parameters": simulation}
