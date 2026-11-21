from tests.support.single_particle_pipeline_runtime_fixtures import (
    Esirkepov_current,
    J_from_rhov,
    _assemble_scalar,
    _assemble_vector,
    _assert_vector_close,
    _empty_electrodynamic_fields,
    _expected_Bz_after_split_update,
    _gather_electric_field,
    _grid_current_work,
    _manual_direct_current_tiles,
    _manual_esirkepov_current_tiles_1d,
    _manual_rho_tiles,
    _one_particle,
    _particle_state,
    _periodic_test_electric_field,
    _runtime_parameters,
    compute_rho,
    empty_tiled_scalar,
    empty_tiled_vector,
    filter_electric_field_for_particles,
    jnp,
    kernel_parameters,
    refresh_tiled_particle_tiles,
    time_loop_electrodynamic,
    unittest,
    update_tiled_particle_positions,
)


class TestSingleParticleStencils(unittest.TestCase):
    def test_single_particle_rho_matches_exact_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(shape_factor=shape_factor)
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, (0.0, 0.2, 0.0))
                    rho = compute_rho(
                        particles,
                        species_config,
                        empty_tiled_scalar(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )
                    expected = _manual_rho_tiles(particles, species_config, static_parameters, dynamic_parameters)

                    self.assertTrue(jnp.allclose(rho, expected, rtol=1.0e-12, atol=1.0e-12))
                    total_charge = jnp.sum(_assemble_scalar(rho, static_parameters)[1:-1, 1:-1, 1:-1])
                    total_charge *= dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
                    self.assertAlmostEqual(float(total_charge), -0.5, places=12)

    def test_single_particle_direct_current_matches_exact_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        u = (0.11, -0.17, 0.07)
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(shape_factor=shape_factor)
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, u)
                    J = J_from_rhov(
                        particles,
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )
                    expected = _manual_direct_current_tiles(particles, species_config, static_parameters, dynamic_parameters)

                    _assert_vector_close(self, J, expected)


    def test_single_particle_esirkepov_current_matches_exact_1d_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        u = (0.08, -0.17, 0.07)
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(
                        shape_factor=shape_factor,
                        current_deposition="esirkepov",
                    )
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, u)
                    J = Esirkepov_current(
                        particles,
                        update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt),
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                        coordinate_velocity=particles.u,
                    )
                    expected = _manual_esirkepov_current_tiles_1d(
                        particles,
                        species_config,
                        static_parameters,
                        dynamic_parameters,
                    )

                    _assert_vector_close(self, J, expected)


if __name__ == "__main__":
    unittest.main()
