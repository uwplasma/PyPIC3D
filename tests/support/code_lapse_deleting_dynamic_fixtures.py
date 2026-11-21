"""Lapse deletion at dynamic PIC initialization and committed metric changes."""

import unittest

import jax
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed") from error
    raise

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    build_bssn_metric, initialize_dynamical_relativity,
    rephase_pic_state, synchronize_particles,
)
from PyPIC3D.relativity.energy_momentum_tensor import compute_particle_energy_momentum
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


jax.config.update("jax_enable_x64", True)


def dynamic_lapse_state(*, varying_lapse=False):
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., tile_shape=(8, 1, 1),
        dt=.02, guard_cells=3, solver="dynamic_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        particle_batch_size=2,
    )
    static = static._replace(lapse_deleting=True, lapse_threshold=.1, metric_backreaction=False)
    static, bssn = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    if varying_lapse:
        phase = 2 * jnp.pi * jnp.arange(8)[:, None, None] / 8
        bssn = bssn._replace(state=bssn.state._replace(lapse=.15 + .1 * jnp.cos(phase)))
    dynamic = dynamic._replace(bssn_parameters=bssn)

    nodes = dynamic.grids.tiled_center_grid[0][0, 0, 0]
    x = jnp.zeros((1, 1, 1, 1, 2, 3)).at[..., 0].set(nodes[jnp.array([3, 7])])
    u = jnp.broadcast_to(jnp.array([.2, .05, 0.]), x.shape)
    particles = TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.array([.4]), jnp.array([1.]), jnp.array([.01]),
                            jnp.ones((1, 3), dtype=bool))
    zero = empty_tiled_vector(static, dynamic)
    fields = (zero, zero, zero, zero[0], zero[0], (zero, zero), None,
              (zero, zero), jnp.asarray(False))
    return static, dynamic, particles, species, fields


class LapseDeletingDynamicTestsFixtures:
    def assert_tree_close(self, actual, expected):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(bool(jnp.allclose(a, b, rtol=2e-12, atol=2e-12)),
                            f"unequal array with shape {a.shape}")
