"""Filtered BSSN sources and raw physical work-energy measurements."""

import unittest

import jax
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from PyPIC3D.diagnostics.gr_work_energy_theorem import sample_energy
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_to_tiles, build_bssn_metric, synchronize_particles,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    electromagnetic_moments,
)
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


def parameters():
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., guard_cells=3,
        solver="dynamic_metric", metric="numerical", shape_factor=1,
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="bilinear", particle_batch_size=2,
    )
    static, bssn = build_bssn_parameters(static, dynamic)
    phase = 2*jnp.pi*jnp.arange(8)[:, None, None]/8
    state = bssn.state._replace(
        conformal_factor=.9 + .15*jnp.cos(phase),
        trace_K=jnp.full_like(phase, .12),
        lapse=1. + .2*jnp.sin(phase),
    )
    return static, dynamic._replace(bssn_parameters=bssn._replace(state=state))


def plasma(dynamic, *, empty=False):
    count = 0 if empty else 1
    x = jnp.zeros((1, 1, 1, count, count, 3))
    if not empty:
        x = x.at[..., 0].set(dynamic.grids.tiled_center_grid[0][0, 0, 0, 5])
    u = jnp.broadcast_to(jnp.array([.6, -.3, .2]), x.shape)
    particles = TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.ones(count), jnp.full((count,), 2.),
                            jnp.full((count,), .3), jnp.ones((count, 3), dtype=bool))
    return particles, species


def binomial(value):
    """The one-dimensional periodic [1,2,1]/4 reference stencil."""
    return (jnp.roll(value, 1, axis=-3) + 2*value + jnp.roll(value, -1, axis=-3))/4


