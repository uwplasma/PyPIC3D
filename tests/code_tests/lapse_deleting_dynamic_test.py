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
    bssn = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
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


class LapseDeletingDynamicTests(unittest.TestCase):
    def assert_tree_close(self, actual, expected):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(bool(jnp.allclose(a, b, rtol=2e-12, atol=2e-12)),
                            f"unequal array with shape {a.shape}")

    def test_initial_sources_and_histories_exclude_low_lapse_particles(self):
        static, dynamic, particles, species, fields = dynamic_lapse_state(varying_lapse=True)
        surviving = particles.active.at[..., 1].set(False)
        reference_particles = particles._replace(active=surviving)
        expected_p, expected_f, expected_d = initialize_dynamical_relativity(
            reference_particles, species, fields,
            static._replace(lapse_deleting=False), dynamic)
        actual_p, actual_f, actual_d = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)

        self.assertTrue(bool(jnp.array_equal(actual_p.active, surviving)))
        self.assert_tree_close(actual_f, expected_f)
        self.assertTrue(bool(jnp.any(actual_f[2][0] != 0.)))
        common = synchronize_particles(actual_p, species, actual_f, static, actual_d)
        actual_sources = compute_particle_energy_momentum(
            common, species, actual_f[6], static, actual_d)
        expected_common = synchronize_particles(
            expected_p, species, expected_f, static, expected_d)
        expected_sources = compute_particle_energy_momentum(
            expected_common, species, expected_f[6], static, expected_d)
        self.assert_tree_close(actual_sources, expected_sources)

    def test_metric_rephase_removes_particles_from_both_common_time_paths(self):
        static, dynamic, particles, species, fields = dynamic_lapse_state()
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)
        common = synchronize_particles(particles, species, fields, static, dynamic)
        phase = 2 * jnp.pi * jnp.arange(8)[:, None, None] / 8
        changed_state = dynamic.bssn_parameters.state._replace(lapse=.15 + .1 * jnp.cos(phase))
        metric = build_bssn_metric(changed_state, static, dynamic)
        surviving = particles.active.at[..., 1].set(False)

        for supplied in (False, True):
            with self.subTest(synchronized_particles=supplied):
                expected = rephase_pic_state(
                    particles._replace(active=surviving), species, fields, metric,
                    static._replace(lapse_deleting=False), dynamic,
                    synchronized_particles=common._replace(active=surviving) if supplied else None)
                advance = jax.jit(lambda p, f: rephase_pic_state(
                    p, species, f, metric, static, dynamic,
                    synchronized_particles=common if supplied else None))
                actual_p, actual_f = advance(particles, fields)
                self.assertTrue(bool(jnp.array_equal(actual_p.active, surviving)))
                self.assert_tree_close(actual_f, expected[1])
                self.assertTrue(bool(jnp.allclose(actual_p.u[..., 0, :], expected[0].u[..., 0, :],
                                                  rtol=2e-12, atol=2e-12)))

    def test_supplied_common_time_state_cannot_reactivate_an_inactive_particle(self):
        static, dynamic, particles, species, fields = dynamic_lapse_state()
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)
        common = synchronize_particles(particles, species, fields, static, dynamic)
        particles = particles._replace(active=particles.active.at[..., 1].set(False))
        changed_state = dynamic.bssn_parameters.state._replace(
            lapse=dynamic.bssn_parameters.state.lapse * .8)
        metric = build_bssn_metric(changed_state, static, dynamic)
        actual = rephase_pic_state(
            particles, species, fields, metric, static, dynamic, synchronized_particles=common)
        expected = rephase_pic_state(
            particles, species, fields, metric, static._replace(lapse_deleting=False), dynamic,
            synchronized_particles=common._replace(active=particles.active))
        self.assertTrue(bool(jnp.array_equal(actual[0].active, particles.active)))
        self.assert_tree_close(actual[1], expected[1])

    def test_full_dynamic_step_uses_the_updated_lapse_before_the_second_pic_half(self):
        static, dynamic, particles, species, fields = dynamic_lapse_state()
        particles = particles._replace(u=jnp.zeros_like(particles.u))
        species = species._replace(charge=jnp.zeros_like(species.charge),
                                   update_x=jnp.zeros_like(species.update_x))
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)

        def prescribed_gauge(state, time):
            lapse = jnp.full_like(state.lapse, jnp.where(time < dynamic.dt / 2, 1., .05))
            return lapse, state.shift, jnp.zeros_like(lapse), jnp.zeros_like(state.shift)

        advance = jax.jit(lambda p, f, d: time_loop_dynamical_relativity(
            p, species, f, static, d, prescribed_gauge=prescribed_gauge))
        updated_p, updated_f, updated_d = advance(particles, fields, dynamic)
        self.assertFalse(bool(jnp.any(updated_p.active)))
        self.assertTrue(bool(jnp.all(updated_p.u == 0.)))
        self.assertTrue(bool(jnp.all(updated_d.bssn_parameters.state.lapse == .05)))
        self.assertAlmostEqual(float(updated_d.bssn_parameters.time), float(dynamic.dt))
        for value in jax.tree.leaves((updated_p, updated_f, updated_d)):
            self.assertTrue(bool(jnp.all(jnp.isfinite(value))))
        second_p, _, _ = advance(updated_p, updated_f, updated_d)
        self.assertFalse(bool(jnp.any(second_p.active)))


if __name__ == "__main__":
    unittest.main()
