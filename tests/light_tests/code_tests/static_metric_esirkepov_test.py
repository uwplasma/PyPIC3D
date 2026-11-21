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
