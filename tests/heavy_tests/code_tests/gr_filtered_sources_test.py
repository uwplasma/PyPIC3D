from tests.support.gr_filtered_sources_runtime_fixtures import (
    build_bssn_metric,
    compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    jnp,
    parameters,
    plasma,
    unittest,
)


class FilteredSourcesTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=2.e-12, atol=2.e-12)),
                        f"actual={actual}, expected={expected}")


    def test_disabled_filter_matches_unfiltered_pic_and_em_sum(self):
        static, dynamic = parameters()
        static = static._replace(current_filter="none")
        particles, species = plasma(dynamic)
        state = dynamic.bssn_parameters.state
        metric = build_bssn_metric(state, static, dynamic)
        moments = tuple(jnp.zeros((3, 3) + state.conformal_factor.shape) for _ in range(3))
        pic = compute_particle_energy_momentum(particles, species, metric, static, dynamic)
        em = compute_electromagnetic_energy_momentum(moments, state)
        actual = compute_energy_momentum_tensor(particles, species, moments, state, static, dynamic)
        for value, particle, field in zip(actual, pic, em):
            self.assertClose(value, particle+field)


if __name__ == "__main__":
    unittest.main()
