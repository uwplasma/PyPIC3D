from tests.support.code_lapse_deleting_dynamic_fixtures import (
    LapseDeletingDynamicTestsFixtures,
    build_bssn_metric,
    dynamic_lapse_state,
    initialize_dynamical_relativity,
    jnp,
    rephase_pic_state,
    synchronize_particles,
    unittest,
)


class LapseDeletingDynamicTests(LapseDeletingDynamicTestsFixtures, unittest.TestCase):


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


if __name__ == "__main__":
    unittest.main()
