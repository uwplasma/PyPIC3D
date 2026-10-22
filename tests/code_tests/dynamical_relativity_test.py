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
    tiles_to_bssn,
)
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import (
    advance_pic_half_step,
    time_loop_dynamical_relativity,
)
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update("jax_enable_x64", True)


def coupled_state(*, dt=0.02, gravity_G=0.0, particles=False, boundaries=(0, 0, 0), varying_geometry=False):
    """Small uniform 1D plasma; empty storage supplies the EM-only cases."""
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        tile_shape=(8, 1, 1), dt=dt, guard_cells=3,
        solver="dynamic_metric", metric="flat_cartesian",
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="none", boundary_conditions=boundaries, particle_batch_size=8,
    )
    bssn = build_bssn_parameters(static, dynamic, gravity_G=gravity_G)
    bssn = bssn._replace(evolution_parameters=bssn.evolution_parameters._replace(nu=0.))
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


class DynamicalRelativityTests(unittest.TestCase):
    def assert_tree_close(self, actual, expected, *, atol=2e-12):
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            self.assertTrue(bool(jnp.allclose(a, b, rtol=2e-12, atol=atol)),
                            f"unequal array with shape {a.shape}")

    def test_half_step_agrees_with_prescribed_metric_pic(self):
        static, dynamic, particles, species, fields = coupled_state(particles=True)
        half = dynamic._replace(dt=fields[7][3])
        ordinary = fields[:7] + (fields[7][:2], fields[8])
        expected_particles, expected_fields = jax.jit(
            lambda p, f: time_loop_static_metric(p, species, f, static, half)
        )(particles, ordinary)
        actual_particles, actual_fields = jax.jit(
            lambda p, f: advance_pic_half_step(p, species, f, static, dynamic)
        )(particles, fields)
        self.assert_tree_close(actual_particles, expected_particles)
        self.assert_tree_close(actual_fields[:7], expected_fields[:7])
        self.assert_tree_close(actual_fields[7][:2], expected_fields[7])
        self.assertFalse(bool(actual_fields[8]))

    def test_zero_gravity_matches_two_pic_half_steps(self):
        static, dynamic, particles, species, fields = coupled_state(particles=True)
        advance = jax.jit(lambda p, f, d: advance_pic_half_step(p, species, f, static, d))
        expected_p, expected_f = advance(particles, fields, dynamic)
        expected_p, expected_f = advance(expected_p, expected_f, dynamic)
        actual_p, actual_f, actual_d = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        self.assert_tree_close(actual_p, expected_p)
        self.assert_tree_close(actual_f, expected_f)
        self.assert_tree_close(actual_d.bssn_parameters.state, dynamic.bssn_parameters.state)
        self.assertAlmostEqual(float(actual_d.bssn_parameters.time), float(dynamic.dt))

    def test_vacuum_gravity_matches_standalone_projected_rk4(self):
        static, dynamic, particles, species, fields = coupled_state(gravity_G=1.)
        bssn = dynamic.bssn_parameters
        state = bssn.state._replace(trace_K=jnp.full_like(bssn.state.trace_K, .08))
        dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))
        zero = tuple(jnp.zeros_like(q) for q in fields[0])
        fields = (zero, zero, zero) + fields[3:7] + (
            (zero, zero, zero, fields[7][3], None), fields[8],
        )
        expected = rk4_step(state, bssn.evolution_parameters)
        actual_p, actual_f, actual_d = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        self.assert_tree_close(actual_d.bssn_parameters.state, expected)
        self.assert_tree_close(actual_f[:3], (zero, zero, zero))
        self.assertEqual(actual_p.active.size, 0)

    def test_identical_metric_preserves_every_history_sample(self):
        static, dynamic, particles, species, fields = coupled_state(particles=True)
        p, f = jax.jit(lambda p, f: rephase_pic_state(
            p, species, f, f[6], static, dynamic,
        ))(particles, fields)
        for actual, expected in zip(jax.tree_util.tree_leaves((p, f)),
                                    jax.tree_util.tree_leaves((particles, fields))):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_esirkepov_gauss_constraint_survives_gravity_and_tile_crossings(self):
        static, dynamic, particles, species, fields = coupled_state(
            particles=True, gravity_G=.03, dt=.04,
        )
        # The supported one-device mesh has one tile. Cross both periodic
        # faces, so deposition must use the unwrapped endpoint displacement.
        x = jnp.zeros((1, 1, 1, 1, 2, 3)).at[..., 0].set(
            jnp.asarray([-3.999, 3.999]).reshape(1, 1, 1, 1, 2)
        )
        u = jnp.zeros_like(x).at[..., 0].set(
            jnp.asarray([-.8, .8]).reshape(1, 1, 1, 1, 2)
        )
        active = jnp.ones(x.shape[:-1], bool)
        particles = TiledParticles(x, u, active)
        initial = gauss_residual(particles, species, fields, static, dynamic)
        half_p, half_f = jax.jit(
            lambda p, f: advance_pic_half_step(p, species, f, static, dynamic)
        )(particles, fields)
        half_residual = gauss_residual(half_p, species, half_f, static, dynamic)
        self.assertTrue(bool(jnp.allclose(half_residual, initial, rtol=0., atol=2e-13)))
        final_p, final_f, final_d = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        final_residual = gauss_residual(final_p, species, final_f, static, final_d)
        self.assertTrue(bool(jnp.allclose(final_residual, initial, rtol=0., atol=2e-13)))
        self.assertEqual(int(jnp.sum(final_p.active)), 2)
        self.assertFalse(bool(final_f[8]))

    def test_vacuum_conducting_boundary_matches_fixed_metric_half_steps(self):
        static, dynamic, particles, species, fields = coupled_state(
            boundaries=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        ordinary = fields[:7] + (fields[7][:2], fields[8])
        half = dynamic._replace(dt=fields[7][3])
        fixed_step = jax.jit(lambda p, f: time_loop_static_metric(p, species, f, static, half))
        expected_p, expected_f = fixed_step(particles, ordinary)
        expected_p, expected_f = fixed_step(expected_p, expected_f)
        actual_p, actual_f, _ = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        self.assert_tree_close(actual_p, expected_p)
        self.assert_tree_close(actual_f[:7], expected_f[:7])
        self.assert_tree_close(actual_f[7][:2], expected_f[7])

    def test_charged_time_refinement_with_lapse_shift_and_spatial_metric(self):
        static, _, _, species, _ = coupled_state(particles=True)

        def gauge(state, time):
            phase = 2 * jnp.pi * jnp.arange(8)[:, None, None] / 8
            profile = 1. + .05 * jnp.cos(phase)
            lapse = (1. + .2 * jnp.sin(time)) * profile
            shift = jnp.zeros_like(state.shift).at[0].set(
                .1 * jnp.sin(2 * time) + .02 * jnp.sin(phase))
            shift = shift.at[1].set(.04 * jnp.sin(time))
            lapse_dot = .2 * jnp.cos(time) * profile
            shift_dot = jnp.zeros_like(state.shift).at[0].set(.2 * jnp.cos(2 * time))
            shift_dot = shift_dot.at[1].set(.04 * jnp.cos(time))
            return lapse, shift, lapse_dot, shift_dot

        def evolve(particles, fields, dynamic, steps):
            def step(_, carry):
                p, f, d = carry
                return time_loop_dynamical_relativity(
                    p, species, f, static, d, prescribed_gauge=gauge,
                )
            p, f, d = jax.lax.fori_loop(0, steps, step, (particles, fields, dynamic))
            common = synchronize_particles(p, species, f, static, d)
            # Compare physical common-time quantities, not dt-dependent history.
            return common.x, common.u, f[0], f[7][2], d.bssn_parameters.state

        evolve = jax.jit(evolve)
        results = []
        for steps in (8, 16, 32):
            _, dynamic, particles, _, fields = coupled_state(
                particles=True, dt=.32 / steps, varying_geometry=True)
            results.append(evolve(particles, fields, dynamic, steps))
        errors = []
        for coarse, fine in zip(results, results[1:]):
            errors.append(jnp.sqrt(sum(
                jnp.mean((a-b)**2)
                for a, b in zip(jax.tree_util.tree_leaves(coarse), jax.tree_util.tree_leaves(fine))
            )))
        ratio = float(errors[0] / errors[1])
        self.assertGreater(ratio, 3.2, f"temporal refinement ratio {ratio}")
        self.assertLess(ratio, 4.8, f"temporal refinement ratio {ratio}")
        final_geometry = results[-1][-1]
        self.assertGreater(float(jnp.max(jnp.abs(final_geometry.conformal_factor - 1.))), .01)
        self.assertGreater(float(jnp.max(jnp.abs(final_geometry.shift))), .01)
        self.assertGreater(float(jnp.max(jnp.abs(final_geometry.lapse - 1.))), .01)


if __name__ == "__main__":
    unittest.main()
