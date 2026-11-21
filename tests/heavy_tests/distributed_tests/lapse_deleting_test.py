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
