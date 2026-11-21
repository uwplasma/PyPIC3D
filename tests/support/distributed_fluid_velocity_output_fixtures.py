"""Particle diagnostic velocities stay on the same tiles as their metric."""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.diagnostics.output_adapters import particle_three_velocity
from PyPIC3D.diagnostics.fluid_quantities import compute_velocity_field
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.core import Metric, YeeMetric, contravariant_three_velocity
from PyPIC3D.relativity.interpolate_metric import (
    interpolate_metric, particle_lorentz_factor, safe_inactive_positions,
)
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push


class TestTiledParticleDiagnosticVelocityFixtures:
    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES=2")
        self.static, self.dynamic = kernel_parameters(
            Nx=8, Ny=4, Nz=4, tile_shape=(4, 4, 4), guard_cells=3,
            solver="dynamic_metric", metric="numerical")
        self.sharding = NamedSharding(self.static.field_mesh, P("tile_x", "tile_y", "tile_z"))
        grid = self.dynamic.grids.tiled_center_grid
        x, y, z = jnp.broadcast_arrays(grid[0][..., :, None, None],
                                       grid[1][..., None, :, None],
                                       grid[2][..., None, None, :])
        diagonal = jnp.stack((2.0 + 0.03*x, 1.2 + 0.02*y, 1.8 + 0.04*z), axis=-1)
        gamma = jnp.eye(3)*diagonal[..., None, :]
        metric = Metric(jnp.ones_like(x), jnp.zeros(x.shape + (3,)), gamma,
                        jnp.eye(3)/diagonal[..., None, :], jnp.sqrt(jnp.prod(diagonal, axis=-1)))
        self.metric = jax.tree.map(lambda a: jax.device_put(a, self.sharding), metric)

        positions = jnp.asarray([[-1.7, -0.6, -0.3], [-1.1, 0.1, 0.2], [-0.3, 0.7, 0.6]])
        positions = jnp.broadcast_to(positions, (2, 1, 1, 2, 3, 3))
        positions = positions.at[1, ..., 0].add(2.0)
        momenta = jnp.broadcast_to(jnp.asarray([0.2, -0.3, 0.5]), positions.shape)
        momenta = momenta.at[..., 1, :, :].multiply(1.5)
        active = jnp.ones(positions.shape[:-1], dtype=bool).at[..., 1, 2].set(False)
        # Unused slots deliberately contain NaNs, as they need not hold valid
        # positions or momenta for the physical particle diagnostic.
        positions = jnp.where(active[..., None], positions, jnp.nan)
        momenta = jnp.where(active[..., None], momenta, jnp.nan)
        self.particles = TiledParticles(*(jax.device_put(a, self.sharding)
                                          for a in (positions, momenta, active)))
