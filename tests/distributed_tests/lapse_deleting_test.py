"""Lapse deletion agrees across particle migration and periodic wrapping."""

import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.lapse_deleting import lapse_deleting
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.support.lapse_deleting_fixtures import (
    active_particle_rows,
    interior_line,
    lapse_particles,
    lapse_runtime,
)


@unittest.skipUnless(jax.device_count() >= 2, "Requires two JAX devices")
class TestDistributedLapseDeleting(unittest.TestCase):
    def test_migration_and_wrapping_sample_the_new_owner_lapse(self):
        results = []
        for tile_width in (8, 4):
            static, dynamic, fields, _ = lapse_runtime(tile_width=tile_width)
            # The first two particles cross the tile seam; the others cross periodic ends.
            particles = lapse_particles(
                static, [0.2, -0.2, 4.1, -4.1], [1.0, 2.0, 3.0, 4.0],
                owner_positions=[-0.1, 0.1, 3.9, -3.9])

            def refresh_and_delete(p):
                p, overflow = refresh_tiled_particle_tiles(p, static, dynamic)
                return lapse_deleting(p, fields[6], static, dynamic), overflow

            deleted, overflow = jax.jit(refresh_and_delete)(particles)
            self.assertFalse(bool(overflow))
            self.assertEqual(deleted.x.shape, particles.x.shape)
            self.assertEqual(int(jnp.sum(deleted.active)), 2)
            positions, momenta = active_particle_rows(deleted)
            self.assertTrue(bool(jnp.allclose(positions[:, 0], jnp.array([-3.9, -0.2]), atol=1e-14)))
            self.assertTrue(bool(jnp.array_equal(momenta[:, 0], jnp.array([3.0, 2.0]))))
            results.append((positions, momenta))
        for actual, expected in zip(results[0], results[1]):
            actual, expected = (jax.device_put(value, jax.devices()[0]) for value in (actual, expected))
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_crossing_step_matches_single_device_for_both_current_schemes(self):
        for scheme in ("esirkepov", "GR_direct"):
            with self.subTest(scheme=scheme):
                results = []
                for tile_width in (8, 4):
                    static, dynamic, fields, species = lapse_runtime(tile_width=tile_width, scheme=scheme)
                    particles = lapse_particles(static, [-0.005, -1.5], [1.0, 0.0])
                    deleted, updated = jax.jit(
                        lambda p, f: time_loop_static_metric(p, species, f, static, dynamic)
                    )(particles, fields)
                    self.assertFalse(bool(updated[-1]))
                    self.assertEqual(int(jnp.sum(deleted.active)), 1)
                    positions, momenta = active_particle_rows(deleted)
                    lines = tuple(interior_line(component, static)
                                  for vector in updated[:3] for component in vector)
                    results.append((positions, momenta, lines))
                for actual, expected in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
                    actual, expected = (jax.device_put(value, jax.devices()[0]) for value in (actual, expected))
                    self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-13, atol=1e-13)))


if __name__ == "__main__":
    unittest.main()
