"""One-tile numerical and physical checks."""

from tests.support.dark_boundaries_fixtures import (
    DarkBoundariesFixtures,
    evolve,
    initialize_dark_photon_fields,
    initialize_dark_pml,
    jnp,
    kernel_parameters,
    load_pml_from_toml,
    np,
    pml_evolve,
    stretch_dark_derivatives,
    synchronized_dark_fields,
    unittest,
)


class TestDarkBoundaries(DarkBoundariesFixtures, unittest.TestCase):
    def test_ade_interval_average_and_zero_sigma(self):
        sigma = (jnp.array([0., .3, 20.]),)*3
        d, p = jnp.array([2., 3., 4.]), jnp.array([0., .2, -.1])
        for dt in (.1, -.05):
            result, memory = stretch_dark_derivatives((d,), (p,), sigma, (0,), dt)
            q = np.asarray(sigma[0]) * dt
            expected = np.exp(-q)*np.asarray(p)+np.expm1(-q)*np.asarray(d)
            np.testing.assert_allclose(memory[0], expected, atol=1e-15)
            np.testing.assert_allclose(result[0][1:], -(expected[1:]-p[1:])/q[1:], atol=1e-14)
            self.assertEqual(float(result[0][0]), 2.)


if __name__ == "__main__":
    unittest.main()
