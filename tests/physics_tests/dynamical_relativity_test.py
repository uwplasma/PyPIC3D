from tests.support.dynamical_relativity_runtime_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    TiledParticles,
    coupled_state,
    gauss_residual,
    jax,
    jnp,
    rk4_step,
    synchronize_particles,
    synchronized_magnetic_field,
    time_loop_dynamical_relativity,
    time_loop_static_metric,
    unittest,
)


class DynamicalRelativityTests(unittest.TestCase):
    def assert_tree_close(self, actual, expected, *, atol=2e-12):
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            self.assertTrue(bool(jnp.allclose(a, b, rtol=2e-12, atol=atol)),
                            f"unequal array with shape {a.shape}")

    def test_zero_gravity_matches_two_pic_half_steps(self):
        static, dynamic, particles, species, fields = coupled_state(particles=True)
        half = dynamic._replace(dt=fields[7][3])
        ordinary = fields[:7] + (fields[7][:2], fields[8])
        advance = jax.jit(lambda p, f: time_loop_static_metric(p, species, f, static, half))
        expected_p, midpoint = advance(particles, ordinary)
        expected_p, expected_f = advance(expected_p, midpoint)
        before_second = midpoint[:7] + ((*midpoint[7], fields[7][2], half.dt, None), midpoint[8])
        common_B = synchronized_magnetic_field(before_second, expected_f, static, half)
        expected_f = expected_f[:7] + ((*expected_f[7], common_B, half.dt, None), expected_f[8])
        actual_p, actual_f, actual_d = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        self.assert_tree_close(actual_p, expected_p)
        self.assert_tree_close(actual_f, expected_f)
        self.assert_tree_close(actual_d.bssn_parameters.state, dynamic.bssn_parameters.state)
        self.assertAlmostEqual(float(actual_d.bssn_parameters.time), float(dynamic.dt))

    def test_vacuum_gravity_matches_standalone_projected_rk4(self):
        static, dynamic, particles, species, fields = coupled_state(backreaction=True)
        bssn = dynamic.bssn_parameters
        state = bssn.state._replace(trace_K=jnp.full_like(bssn.state.trace_K, .08))
        dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))
        zero = tuple(jnp.zeros_like(q) for q in fields[0])
        fields = (zero, zero, zero) + fields[3:7] + (
            (zero, zero, zero, fields[7][3], None), fields[8],
        )
        expected = rk4_step(state, static.bssn_evolution_parameters)
        actual_p, actual_f, actual_d = jax.jit(
            lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
        )(particles, fields, dynamic)
        self.assert_tree_close(actual_d.bssn_parameters.state, expected)
        self.assert_tree_close(actual_f[:3], (zero, zero, zero))
        self.assertEqual(actual_p.active.size, 0)

    def test_esirkepov_gauss_constraint_survives_gravity_and_periodic_crossings(self):
        static, dynamic, particles, species, fields = coupled_state(
            particles=True, backreaction=True, dt=.04,
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
        half = dynamic._replace(dt=fields[7][3])
        ordinary = fields[:7] + (fields[7][:2], fields[8])
        half_p, half_f = jax.jit(
            lambda p, f: time_loop_static_metric(p, species, f, static, half)
        )(particles, ordinary)
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
