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


class TestTiledParticleDiagnosticVelocity(unittest.TestCase):
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

    def test_nonuniform_metric_matches_analytic_and_serial_particle_velocities(self):
        velocity = particle_three_velocity(self.particles, self.metric, self.static, self.dynamic)
        x, u, active = self.particles
        diagonal = jnp.stack((2.0 + 0.03*x[..., 0], 1.2 + 0.02*x[..., 1],
                              1.8 + 0.04*x[..., 2]), axis=-1)
        expected = (u/diagonal) / jnp.sqrt(1.0 + jnp.sum(u*u/diagonal, axis=-1))[..., None]
        expected = jnp.where(active[..., None], expected, 0.0)
        self.assertTrue(bool(jnp.allclose(velocity, expected, atol=2.e-14, rtol=2.e-14)))

        for tile in range(2):
            grid = tuple(a[tile, 0, 0] for a in self.dynamic.grids.tiled_center_grid)
            local_active = active[tile, 0, 0]
            local_x = safe_inactive_positions(x[tile, 0, 0], local_active, grid, (True, True, True), 3)
            sampled = interpolate_metric(jax.tree.map(lambda a: a[tile, 0, 0], self.metric),
                                         local_x, grid, "numerical", (True, True, True),
                                         (3, 3, 3), derivatives=False)
            serial = contravariant_three_velocity(
                jnp.where(local_active[..., None], u[tile, 0, 0], 0.0), sampled.gamma_inv)
            serial = jnp.where(local_active[..., None], serial, 0.0)
            self.assertTrue(bool(jnp.allclose(velocity[tile, 0, 0], serial, atol=2.e-14, rtol=2.e-14)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(velocity))))

    def test_compiled_sampling_has_no_global_metric_communication(self):
        sample = jax.jit(lambda particles, metric: particle_three_velocity(
            particles, metric, self.static, self.dynamic))
        executable = sample.lower(self.particles, self.metric).compile()
        velocity = executable(self.particles, self.metric)
        self.assertEqual(velocity.sharding.mesh, self.static.field_mesh)
        self.assertEqual(velocity.sharding.spec[:3], ("tile_x", "tile_y", "tile_z"))
        self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1) for shard in velocity.addressable_shards))
        hlo = executable.as_text()
        for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
            self.assertNotIn(collective, hlo)

    def test_velocity_deposition_keeps_local_fields_after_particle_sampling(self):
        species = SpeciesConfig(jnp.ones(2), jnp.ones(2), jnp.asarray([1.0, 2.0]),
                                jnp.ones((2, 3), dtype=bool))

        def output_velocity(particles, metric):
            particles = particles._replace(u=particle_three_velocity(
                particles, metric, self.static, self.dynamic))
            return tuple(compute_velocity_field(
                particles, metric.lapse, direction, self.static, self.dynamic,
                species_config=species) for direction in range(3))

        executable = jax.jit(output_velocity).lower(self.particles, self.metric).compile()
        output = executable(self.particles, self.metric)
        for component in output:
            self.assertEqual(component.sharding.mesh, self.static.field_mesh)
            self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1)
                                for shard in component.addressable_shards))
            self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
        # Deposition folds and refreshes neighbor halos. It must never exchange
        # complete metric or velocity fields across the device mesh.
        hlo = executable.as_text()
        self.assertIn("collective-permute", hlo)
        for collective in ("all-gather", "all-reduce", "all-to-all"):
            self.assertNotIn(collective, hlo)

    def test_lorentz_factor_uses_the_local_nonuniform_metric(self):
        metric = YeeMetric((self.metric,)*3, (self.metric,)*3, self.metric, self.metric)
        sample = jax.jit(lambda particles, metric: particle_lorentz_factor(
            particles, metric, self.static, self.dynamic))
        executable = sample.lower(self.particles, metric).compile()
        gamma = executable(self.particles, metric)
        x, u, active = self.particles
        diagonal = jnp.stack((2.0 + 0.03*x[..., 0], 1.2 + 0.02*x[..., 1],
                              1.8 + 0.04*x[..., 2]), axis=-1)
        expected = jnp.where(active, jnp.sqrt(1.0 + jnp.sum(u*u/diagonal, axis=-1)), 1.0)
        self.assertTrue(bool(jnp.allclose(gamma, expected, atol=2.e-14, rtol=2.e-14)))
        self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1) for shard in gamma.addressable_shards))
        for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
            self.assertNotIn(collective, executable.as_text())

    def test_hybrid_push_gathers_only_the_metric_and_fields_on_its_device(self):
        metric = YeeMetric((self.metric,)*3, (self.metric,)*3, self.metric, self.metric)
        D = tuple((i+1)*0.02*self.metric.sqrt_gamma for i in range(3))
        B = tuple((i-1)*0.03*self.metric.sqrt_gamma for i in range(3))
        species = SpeciesConfig(jnp.asarray([-1.0, 1.0]), jnp.asarray([1.0, 2.0]),
                                jnp.ones(2), jnp.asarray([[True, True, True], [True, False, True]]))
        particles = self.particles._replace(active=self.particles.active.at[1, ..., 1, :].set(False))
        static = self.static._replace(particle_batch_size=2, particle_pusher="hybrid_boris_geodesic")
        for advance in (False, True):
            with self.subTest(advance_positions=advance):
                push = jax.jit(lambda p, D, B, metric: hybrid_boris_geodesic_push(
                    p, species, D, B, metric, static, self.dynamic, advance_positions=advance))
                executable = push.lower(particles, D, B, metric).compile()
                full, half = executable(particles, D, B, metric)
                for result in (full, half):
                    self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1)
                                        for shard in result.x.addressable_shards))
                    self.assertTrue(bool(jnp.array_equal(result.active, particles.active)))
                    for old, new in ((particles.x, result.x), (particles.u, result.u)):
                        self.assertTrue(bool(jnp.array_equal(old[~particles.active], new[~particles.active],
                                                            equal_nan=True)))
                        self.assertTrue(bool(jnp.all(jnp.isfinite(new[particles.active]))))
                    if not advance:
                        self.assertTrue(bool(jnp.array_equal(result.x, particles.x, equal_nan=True)))
                for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
                    self.assertNotIn(collective, executable.as_text())

if __name__ == "__main__":
    unittest.main()
