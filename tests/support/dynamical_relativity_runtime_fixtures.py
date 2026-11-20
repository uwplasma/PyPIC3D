"""Conservation, staggering, and temporal accuracy of the BSSN/PIC split."""

import unittest

import jax
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from JAX_BSSN.evolution.time_evolve import rk4_step
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    initialize_dynamical_relativity,
    rephase_pic_state,
    synchronize_particles,
    synchronized_magnetic_field,
    tiles_to_bssn,
)
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import (
    time_loop_dynamical_relativity,
)
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update("jax_enable_x64", True)


def coupled_state(*, dt=0.02, backreaction=False, particles=False, boundaries=(0, 0, 0), varying_geometry=False):
    """Small uniform 1D plasma; empty storage supplies the EM-only cases."""
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        tile_shape=(8, 1, 1), dt=dt, guard_cells=3,
        solver="dynamic_metric", metric="flat_cartesian",
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="none", boundary_conditions=boundaries, particle_batch_size=8,
    )
    static = static._replace(metric_backreaction=backreaction)
    static, bssn = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    if varying_geometry:
        phase = 2 * jnp.pi * jnp.arange(8)[:, None, None] / 8
        q = bssn.state
        q = q._replace(
            conformal_factor=1. + .02 * jnp.sin(phase),
            trace_K=jnp.full_like(q.trace_K, .12),
            lapse=1. + .05 * jnp.cos(phase),
            shift=q.shift.at[0].set(.02 * jnp.sin(phase)),
        )
        bssn = bssn._replace(state=q)
    dynamic = dynamic._replace(bssn_parameters=bssn)

    zero = empty_tiled_vector(static, dynamic)
    D = tuple(jnp.full_like(zero[i], (0.07, -0.04, 0.02)[i]) for i in range(3))
    B = tuple(jnp.full_like(zero[i], (0.03, 0.02, -0.01)[i]) for i in range(3))
    if particles:
        x = jnp.zeros((1, 1, 1, 1, 8, 3))
        centers = dynamic.grids.tiled_center_grid[0][..., 3:11]
        centers = centers.reshape((1, 1, 1, 1, 8))
        x = x.at[..., 0].set(centers + 0.23)
        u = jnp.broadcast_to(jnp.asarray((0.25, 0.13, -0.07)), x.shape)
        particle_state = TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool))
        species = SpeciesConfig(
            jnp.asarray([0.5]), jnp.asarray([1.]), jnp.asarray([0.01]),
            jnp.ones((1, 3), dtype=bool),
        )
    else:
        x = jnp.zeros((1, 1, 1, 0, 0, 3))
        particle_state = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
        v = jnp.zeros((0,))
        species = SpeciesConfig(v, v, v, jnp.zeros((0, 3), dtype=bool))
    fields = (D, B, zero, zero[0], zero[0], (zero, zero), None, (D, B), jnp.asarray(False))
    particle_state, fields, dynamic = initialize_dynamical_relativity(
        particle_state, species, fields, static, dynamic,
    )
    return static, dynamic, particle_state, species, fields


def gauss_residual(particles, species, fields, static, dynamic):
    rho = compute_rho(particles, species, fields[3], static, dynamic)
    divergence = sum(
        (fields[0][i] - jnp.roll(fields[0][i], 1, axis=3+i)) / spacing
        for i, spacing in enumerate((dynamic.dx, dynamic.dy, dynamic.dz))
    )
    return tiles_to_bssn(divergence - 4 * jnp.pi * rho, static)


