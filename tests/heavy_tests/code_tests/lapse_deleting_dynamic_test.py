from tests.support.code_lapse_deleting_dynamic_fixtures import (
    LapseDeletingDynamicTestsFixtures,
    build_bssn_metric,
    compute_particle_energy_momentum,
    dynamic_lapse_state,
    initialize_dynamical_relativity,
    jax,
    jnp,
    rephase_pic_state,
    synchronize_particles,
    time_loop_dynamical_relativity,
    unittest,
)


class LapseDeletingDynamicTests(LapseDeletingDynamicTestsFixtures, unittest.TestCase):

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
