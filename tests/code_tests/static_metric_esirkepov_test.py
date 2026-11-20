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


class TestGRESirkepovConfiguration(GRESirkepovConfigurationFixtures, unittest.TestCase):
    def test_encoder_accepts_shared_esirkepov(self):
        self.assertEqual(_encode_current_calculation("esirkepov"), "esirkepov")

    def test_encoder_still_maps_the_existing_schemes(self):
        self.assertEqual(_encode_current_calculation("GR_direct_deposition"), "GR_direct")
        self.assertEqual(_encode_current_calculation("esirkepov"), "esirkepov")
        self.assertEqual(_encode_current_calculation("j_from_rhov"), "direct")

    def test_encoder_rejects_an_unknown_scheme(self):
        with self.assertRaises(ValueError):
            _encode_current_calculation("not_a_scheme")

    def test_static_metric_accepts_the_new_scheme(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(self._static_config(), dynamic_config)

    def test_static_metric_still_accepts_direct_deposition(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(
            self._static_config(current_calculation="GR_direct_deposition"),
            dynamic_config,
        )

    def test_all_electromagnetic_solvers_accept_shared_esirkepov(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        for solver in ("electrodynamic_yee", "dark_matter_yee", "static_metric"):
            pusher = "hybrid_boris_geodesic" if solver == "static_metric" else "boris"
            _validate_tiled_yee_configuration(
                self._static_config(solver=solver, particle_pusher=pusher), dynamic_config,
            )

    def test_removed_configuration_name_points_to_shared_scheme(self):
        with self.assertRaisesRegex(ValueError, "Unsupported current_calculation.*'esirkepov'"):
            _encode_current_calculation("GR_esirkepov")

    def test_current_only_filtering_is_rejected_outside_matched_gr(self):
        """Ordinary Yee has no matched charge/current filtering contract."""

        for solver in ("electrodynamic_yee", "dark_matter_yee"):
            for filter_name in ("digital", "bilinear"):
                with self.subTest(solver=solver, filter=filter_name):
                    with self.assertRaisesRegex(ValueError, "use filter_j='none'"):
                        _validate_current_filter_contract(
                            self._static_config(solver=solver, filter_j=filter_name)
                        )

    def test_periodic_cartesian_gr_accepts_matched_filtering(self):
        for solver in ("static_metric", "dynamic_metric"):
            for filter_name in ("digital", "bilinear"):
                _validate_current_filter_contract(
                    self._static_config(solver=solver, filter_j=filter_name))

    def test_unfiltered_configuration_is_accepted(self):
        _validate_current_filter_contract(self._static_config())

    def test_direct_deposition_may_still_be_filtered(self):
        _validate_current_filter_contract(
            self._static_config(current_calculation="GR_direct_deposition", filter_j="digital")
        )


if __name__ == "__main__":
    unittest.main()
