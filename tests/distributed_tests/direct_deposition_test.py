"""Focused cross-device regression tests."""

from tests.support.direct_deposition_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    DirectDepositionFixtures,
    J_from_rhov,
    assemble_tiled_vector_field,
    jnp,
    kernel_parameters_from_values,
    unittest,
)


class TestDirectDeposition(DirectDepositionFixtures, unittest.TestCase):
    def test_tiled_direct_deposition_matches_quadratic_with_two_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((0, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((1, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((1, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((1, 1, 1), 0, 2, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_matches_quadratic_saved_style_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_bilinear_matches_quadratic_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")


    def test_tiled_direct_deposition_returns_only_local_current_tiles(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 1.0}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))

        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="none",
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))


    def test_tiled_direct_deposition_matches_J_from_rhov_for_dummy_species(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=2,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((0, 1, 1), 1, 0, (-1.65, 1.15, 0.35), (-0.1, 0.3, 0.1), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 0, 0), 1, 0, (0.15, -0.75, -0.45), (0.2, -0.05, 0.05), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 1, 0, (1.75, 0.45, 0.85), (-0.25, 0.15, -0.2), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(
            charges=[-1.0, 2.0],
            masses=[1.0, 4.0],
            weights=[0.5, 0.25],
        )

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_public_J_from_rhov_dispatches_tiled_particles_to_tile_local_current(self):
        parameter_set = self._build_parameter_values()
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dynamic_values = {"C": 3.0e8, "alpha": 0.6}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(current_filter="digital")

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            parameter_set,
            tile_shape,
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="digital",
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))


    def test_tiled_direct_deposition_digital_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="digital", alpha=0.6)


    def test_tiled_direct_deposition_bilinear_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=2,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((1, 0, 0), 0, 0, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((1, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((2, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((2, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")


    def test_tiled_direct_deposition_respects_active_mask(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), False),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), False),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_periodic_boundary_crossing(self):
        parameter_set = self._build_parameter_values(Nx=10, Ny=1, Nz=1, dt=0.0)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-parameter_set["x_wind"] / 2 - 0.2 * parameter_set["dx"], 0.0, 0.0), (-0.25, -0.2, 0.15), True),
                ((4, 0, 0), 0, 0, (parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], 0.0, 0.0), (0.5, 0.1, 0.0), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_matches_J_from_rhov_for_conducting_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING},
        )
        parameter_set["particle_boundary_conditions"] = {
            "x": BC_CONDUCTING,
            "y": BC_CONDUCTING,
            "z": BC_CONDUCTING,
        }
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.5, 0.1, -0.15),
                    True,
                ),
                (
                    (3, 1, 0),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], 0.0, -parameter_set["z_wind"] / 2 + 0.1 * parameter_set["dz"]),
                    (-0.25, -0.2, 0.35),
                    True,
                ),
                (
                    (2, 1, 1),
                    0,
                    0,
                    (0.0, parameter_set["y_wind"] / 2 - 0.1 * parameter_set["dy"], parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]),
                    (0.15, 0.3, -0.1),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_matches_J_from_rhov_for_mixed_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_PERIODIC},
        )
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.2, 0.0, -0.05),
                    True,
                ),
                ((1, 0, 0), 0, 0, (-0.5, -0.25, -parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]), (0.05, -0.2, 0.1), True),
                ((2, 1, 1), 0, 0, (0.5, 0.25, parameter_set["z_wind"] / 2 + 0.2 * parameter_set["dz"]), (0.3, 0.1, -0.15), True),
                (
                    (3, 1, 1),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 + 0.2 * parameter_set["dx"], parameter_set["y_wind"] / 2 - 0.2 * parameter_set["dy"], 0.25),
                    (-0.1, 0.15, 0.25),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


    def test_tiled_direct_deposition_reduced_dimensions(self):
        parameter_set = self._build_parameter_values(Nx=16, Ny=1, Nz=1, dt=0.02)
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, 0.0, 0.0), (0.2, 0.3, -0.05), True),
                ((2, 0, 0), 0, 0, (0.15, 0.0, 0.0), (-0.1, 0.15, 0.25), True),
                ((3, 0, 0), 0, 0, (1.25, 0.0, 0.0), (0.05, -0.2, 0.1), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


if __name__ == "__main__":
    unittest.main()
