"""Static-metric deletion preserves the crossing-step current deposition."""

import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.support.lapse_deleting_fixtures import (
    active_particle_rows,
    lapse_particles,
    lapse_runtime,
)


class TestLapseDeletingStatic(unittest.TestCase):
    def test_crossing_retains_final_current_and_stops_subsequent_deposition(self):
        for scheme in ("esirkepov", "GR_direct"):
            with self.subTest(scheme=scheme):
                static, dynamic, fields, species = lapse_runtime(scheme=scheme)
                particles = lapse_particles(static, [-0.005], [1.0])
                disabled = static._replace(lapse_deleting=False)

                reference, reference_fields = jax.jit(
                    lambda p, f: time_loop_static_metric(p, species, f, disabled, dynamic)
                )(particles, fields)
                advance = jax.jit(
                    lambda p, f: time_loop_static_metric(p, species, f, static, dynamic)
                )
                deleted, updated = advance(particles, fields)

                # alpha=0.1-0.01*x: the surface is x=0, and the trajectory crosses it.
                reference_x, _ = active_particle_rows(reference)
                self.assertGreater(float(reference_x[0, 0]), 0.0)
                self.assertFalse(bool(jnp.any(deleted.active)))
                self.assertEqual(deleted.x.shape, particles.x.shape)
                self.assertGreater(float(jnp.max(jnp.abs(updated[2][0]))), 0.0)
                for actual, expected in zip(jax.tree.leaves(updated[:3]),
                                            jax.tree.leaves(reference_fields[:3])):
                    self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-13, atol=1e-13)))

                rho = compute_rho(deleted, species, fields[3], static, dynamic)
                self.assertTrue(bool(jnp.all(rho == 0.0)))
                next_particles, next_fields = advance(deleted, updated)
                self.assertFalse(bool(jnp.any(next_particles.active)))
                self.assertFalse(bool(next_fields[-1]))
                for current in next_fields[2]:
                    self.assertTrue(bool(jnp.all(current == 0.0)))

    def test_disabled_boundary_and_unreached_cut_give_identical_steps(self):
        for scheme in ("esirkepov", "GR_direct"):
            with self.subTest(scheme=scheme):
                static, dynamic, fields, species = lapse_runtime(scheme=scheme)
                particles = lapse_particles(static, [-1.5], [0.2])
                results = []
                # A disabled cut ignores even a threshold above every particle lapse.
                for parameters in (static, static._replace(lapse_deleting=False, lapse_threshold=2.0)):
                    results.append(jax.jit(
                        lambda p, f: time_loop_static_metric(p, species, f, parameters, dynamic)
                    )(particles, fields))
                for actual, expected in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
                    self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_lapse_cut_coexists_with_outer_reflecting_and_absorbing_boundaries(self):
        for scheme in ("esirkepov", "GR_direct"):
            for boundary in (1, 2):
                with self.subTest(scheme=scheme, boundary=boundary):
                    static, dynamic, fields, species = lapse_runtime(
                        scheme=scheme, boundary=boundary, lapse_slope=0.01, dt=0.5)
                    # The negative-x particle starts inside the cut; the other exits x=4.
                    particles = lapse_particles(static, [-1.5, 3.98], [0.0, 1.0])
                    result, updated = jax.jit(
                        lambda p, f: time_loop_static_metric(p, species, f, static, dynamic)
                    )(particles, fields)
                    self.assertFalse(bool(updated[-1]))
                    self.assertEqual(int(jnp.sum(result.active)), 1 if boundary == 1 else 0)
                    if boundary == 1:
                        positions, momenta = active_particle_rows(result)
                        self.assertGreater(float(positions[0, 0]), 3.0)
                        self.assertLess(float(positions[0, 0]), 4.0)
                        self.assertLess(float(momenta[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
