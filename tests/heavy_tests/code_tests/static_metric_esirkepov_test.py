"""One-tile API contracts."""

from tests.support.static_metric_esirkepov_fixtures import (
    DepositionDispatchFixtures,
    Esirkepov_current,
    GRESirkepovConfigurationFixtures,
    GRESirkepovConventionsFixtures,
    TiledParticles,
    _encode_current_calculation,
    _unit_species,
    _validate_current_filter_contract,
    _validate_tiled_yee_configuration,
    empty_tiled_vector,
    jnp,
    unittest,
)


class TestGRESirkepovConventions(GRESirkepovConventionsFixtures, unittest.TestCase):
    def test_frozen_axes_deposit_no_current(self):
        static_parameters, dynamic_parameters = self._flat_setup()
        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        displaced = position + jnp.asarray([(0.31, 0.22, -0.18)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.zeros_like(position)
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        particles_old = TiledParticles(x=position, u=velocity, active=active)
        particles_new = TiledParticles(x=displaced, u=velocity, active=active)
        template = empty_tiled_vector(static_parameters, dynamic_parameters)

        def deposit(update_x):
            return Esirkepov_current(
                particles_old, particles_new,
                _unit_species()._replace(update_x=jnp.asarray([update_x])),
                template, static_parameters, dynamic_parameters,
            )

        full = deposit([True, True, True])
        masked = deposit([False, True, False])
        disabled = deposit([False, False, False])

        self.assertGreater(float(jnp.max(jnp.abs(full[1]))), 0.0)
        self.assertTrue(bool(jnp.allclose(masked[0], 0.0)))
        self.assertTrue(bool(jnp.allclose(masked[2], 0.0)))
        for component in disabled:
            self.assertTrue(bool(jnp.allclose(component, 0.0)))


class TestDepositionDispatch(DepositionDispatchFixtures, unittest.TestCase):
    def test_both_schemes_jit_and_stay_finite(self):
        for current_deposition in ("esirkepov", "GR_direct"):
            with self.subTest(current_deposition=current_deposition):
                fields = self._one_jitted_step(current_deposition)
                for component in fields[0]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
                for component in fields[2]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))

    def test_the_selector_actually_switches_kernels(self):
        """
        Guards against the branch being silently ignored: the two schemes
        deposit genuinely different currents for the same particle state.
        """

        esirkepov = self._one_jitted_step("esirkepov")[2]
        direct = self._one_jitted_step("GR_direct")[2]
        self.assertGreater(float(jnp.max(jnp.abs(esirkepov[0]))), 0.0)
        self.assertGreater(float(jnp.max(jnp.abs(direct[0]))), 0.0)
        differs = any(
            not bool(jnp.allclose(esirkepov[i], direct[i])) for i in range(3)
        )
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
