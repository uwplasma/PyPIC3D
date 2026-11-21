from tests.support.single_particle_pipeline_runtime_fixtures import (
    _assert_vector_close,
    _manual_rho_tiles,
    _one_particle,
    _periodic_test_electric_field,
    _runtime_parameters,
    compute_rho,
    empty_tiled_scalar,
    filter_electric_field_for_particles,
    jnp,
    unittest,
    yee_push_fields,
)


class TestSingleParticleStencils(unittest.TestCase):
    def test_coupling_filter_leaves_none_external_and_magnetic_fields_unchanged(self):
        static_none_06, dynamic_none_06 = _runtime_parameters(current_filter="none", alpha=0.6)
        static_none_10, dynamic_none_10 = _runtime_parameters(current_filter="none", alpha=1.0)
        static_digital, dynamic_digital = _runtime_parameters(current_filter="digital", alpha=0.6)

        E = _periodic_test_electric_field(static_none_06, dynamic_none_06)
        B = tuple(0.3 * component for component in E)
        external_E = tuple(0.2 * component for component in E)
        external_B = tuple(-0.4 * component for component in B)

        coupling_none_06 = filter_electric_field_for_particles(
            E,
            static_none_06,
            dynamic_none_06,
        )
        coupling_none_10 = filter_electric_field_for_particles(
            E,
            static_none_10,
            dynamic_none_10,
        )
        coupling_digital = filter_electric_field_for_particles(
            E,
            static_digital,
            dynamic_digital,
        )
        push_E, push_B = yee_push_fields(
            E,
            B,
            (external_E, external_B),
            static_digital,
            dynamic_digital,
        )

        _assert_vector_close(self, coupling_none_06, E)
        _assert_vector_close(self, coupling_none_10, E)
        self.assertTrue(any(
            not jnp.allclose(filtered, unfiltered, rtol=1.0e-12, atol=1.0e-12)
            for filtered, unfiltered in zip(coupling_digital, E)
        ))
        _assert_vector_close(
            self,
            tuple(total - filtered for total, filtered in zip(push_E, coupling_digital)),
            external_E,
        )
        _assert_vector_close(
            self,
            push_B,
            tuple(field + external for field, external in zip(B, external_B)),
        )

    def test_compute_rho_filter_selector_uses_static_current_filter(self):
        x = (1.97, 0.0, 0.0)
        u = (0.0, 0.2, 0.0)
        static_raw_06, dynamic_raw_06 = _runtime_parameters(shape_factor=2, current_filter="none", alpha=0.6)
        static_raw_10, dynamic_raw_10 = _runtime_parameters(shape_factor=2, current_filter="none", alpha=1.0)
        static_digital, dynamic_digital = _runtime_parameters(shape_factor=2, current_filter="digital", alpha=0.6)

        particles_raw_06, species_config = _one_particle(static_raw_06, dynamic_raw_06, x, u)
        particles_raw_10, _ = _one_particle(static_raw_10, dynamic_raw_10, x, u)
        particles_digital, _ = _one_particle(static_digital, dynamic_digital, x, u)

        rho_raw_06 = compute_rho(
            particles_raw_06,
            species_config,
            empty_tiled_scalar(static_raw_06, dynamic_raw_06),
            static_raw_06,
            dynamic_raw_06,
        )
        rho_raw_10 = compute_rho(
            particles_raw_10,
            species_config,
            empty_tiled_scalar(static_raw_10, dynamic_raw_10),
            static_raw_10,
            dynamic_raw_10,
        )
        rho_digital = compute_rho(
            particles_digital,
            species_config,
            empty_tiled_scalar(static_digital, dynamic_digital),
            static_digital,
            dynamic_digital,
        )
        expected_digital = _manual_rho_tiles(particles_digital, species_config, static_digital, dynamic_digital)

        self.assertTrue(jnp.allclose(rho_raw_06, rho_raw_10, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(rho_digital, expected_digital, rtol=1.0e-12, atol=1.0e-12))
        self.assertGreater(float(jnp.max(jnp.abs(rho_digital - rho_raw_06))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
