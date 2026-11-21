"""One-tile API contracts."""

from tests.support.rho_fixtures import (
    BC_ABSORBING,
    BC_PERIODIC,
    TiledRhoFixtures,
    assemble_tiled_scalar_field,
    compute_rho,
    digital_filter,
    ghost_cells,
    jnp,
    kernel_parameters_from_values,
    tile_scalar_field,
    unittest,
)


class TestTiledRho(TiledRhoFixtures, unittest.TestCase):
    def test_tiled_rho_digital_filter_depends_on_alpha(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        parameter_set["current_filter"] = "digital"
        simulation_parameters = self._one_tile_parameters(parameter_set)
        particles = self._particles(parameter_set)

        rho_tiles_alpha_10, rho_alpha_10 = self._deposit_and_assemble(
            particles,
            parameter_set,
            simulation_parameters,
            {"alpha": 1.0},
        )
        rho_tiles_alpha_055, rho_alpha_055 = self._deposit_and_assemble(
            particles,
            parameter_set,
            simulation_parameters,
            {"alpha": 0.55},
        )

        parameter_set = self._parameters_with_tiled_grids(parameter_set, simulation_parameters)
        static_parameters, _dynamic_parameters = kernel_parameters_from_values(parameter_set, {"alpha": 0.55})
        g = int(parameter_set["guard_cells"])
        filtered_reference_tiles = digital_filter(rho_tiles_alpha_10, 0.55, num_guard_cells=g)
        filtered_reference_tiles = ghost_cells.update_tiled_ghost_cells(
            filtered_reference_tiles,
            static_parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        filtered_reference = assemble_tiled_scalar_field(
            filtered_reference_tiles,
            parameter_set,
            parameter_set["tile_shape"],
            num_guard_cells=g,
        )

        self.assertGreater(float(jnp.max(jnp.abs(rho_alpha_055 - rho_alpha_10))), 1.0e-12)
        self.assertTrue(
            jnp.allclose(
                rho_alpha_055,
                filtered_reference,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_compute_rho_uses_particle_boundary_conditions_for_ghost_folding(self):
        dynamic_values = {"alpha": 1.0}
        periodic_parameters = self._build_parameter_values(
            shape_factor=1,
            particle_boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        )
        absorbing_parameters = self._build_parameter_values(
            shape_factor=1,
            particle_boundary_conditions={
                "x": BC_ABSORBING,
                "y": BC_PERIODIC,
                "z": BC_PERIODIC,
            },
        )
        particles = self._particles(periodic_parameters)

        _, periodic_rho = self._deposit_and_assemble(
            particles,
            periodic_parameters,
            self._one_tile_parameters(periodic_parameters),
            dynamic_values,
        )
        _, absorbing_rho = self._deposit_and_assemble(
            particles,
            absorbing_parameters,
            self._one_tile_parameters(absorbing_parameters),
            dynamic_values,
        )

        max_difference = float(jnp.max(jnp.abs(periodic_rho - absorbing_rho)))
        self.assertGreater(max_difference, 1.0e-12)

    def test_compute_rho_uses_current_positions_not_half_step_back_positions(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        dynamic_values = {"alpha": 1.0}
        particles = self._particles(parameter_set)
        zero_velocity_particles = self._zero_species_velocities(self._particles(parameter_set))

        _, rho_with_velocity = self._deposit_and_assemble(particles, parameter_set, self._one_tile_parameters(parameter_set), dynamic_values)
        _, rho_with_zero_velocity = self._deposit_and_assemble(zero_velocity_particles, parameter_set, self._one_tile_parameters(parameter_set), dynamic_values)

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tiled_global_rho_uses_current_positions_not_half_step_back_positions(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        dynamic_values = {"alpha": 1.0}
        particles = self._particles(parameter_set)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, self._one_tile_parameters(parameter_set))
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, parameter_set)
        zero_velocity_tiled_particles = self._zero_tiled_velocities(tiled_particles)
        rho_tiles = tile_scalar_field(self._empty_scalar(parameter_set), parameter_set, parameter_set["tile_shape"], num_guard_cells=int(parameter_set["guard_cells"]))

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        rho_with_velocity_tiles = compute_rho(tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        rho_with_zero_velocity_tiles = compute_rho(zero_velocity_tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        rho_with_velocity = assemble_tiled_scalar_field(rho_with_velocity_tiles, parameter_set, parameter_set["tile_shape"], num_guard_cells=int(parameter_set["guard_cells"]))
        rho_with_zero_velocity = assemble_tiled_scalar_field(rho_with_zero_velocity_tiles, parameter_set, parameter_set["tile_shape"], num_guard_cells=int(parameter_set["guard_cells"]))

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tile_major_rho_uses_current_positions_not_half_step_back_positions(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, self._one_tile_parameters(parameter_set))
        dynamic_values = {"alpha": 1.0}
        particles = self._particles(parameter_set)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, parameter_set)
        zero_velocity_tiled_particles = self._zero_tiled_velocities(tiled_particles)
        rho_tiles = tile_scalar_field(self._empty_scalar(parameter_set), parameter_set, parameter_set["tile_shape"])

        rho_tiles_with_velocity = compute_rho(
            tiled_particles,
            species_config,
            rho_tiles,
            *kernel_parameters_from_values(parameter_set, dynamic_values),
        )
        rho_tiles_with_zero_velocity = compute_rho(
            zero_velocity_tiled_particles,
            species_config,
            rho_tiles,
            *kernel_parameters_from_values(parameter_set, dynamic_values),
        )
        rho_with_velocity = assemble_tiled_scalar_field(rho_tiles_with_velocity, parameter_set, parameter_set["tile_shape"], num_guard_cells=int(parameter_set["guard_cells"]))
        rho_with_zero_velocity = assemble_tiled_scalar_field(rho_tiles_with_zero_velocity, parameter_set, parameter_set["tile_shape"], num_guard_cells=int(parameter_set["guard_cells"]))

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_compute_rho_dispatches_to_tile_major_deposition_for_tiled_particles(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, self._one_tile_parameters(parameter_set))
        dynamic_values = {"alpha": 1.0}
        particles = self._particles(parameter_set)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, parameter_set)
        rho_tiles = tile_scalar_field(
            self._empty_scalar(parameter_set),
            parameter_set,
            parameter_set["tile_shape"],
            num_guard_cells=int(parameter_set["guard_cells"]),
        )

        rho_tiles = compute_rho(
            tiled_particles,
            species_config,
            rho_tiles,
            *kernel_parameters_from_values(parameter_set, dynamic_values),
        )
        rho_from_tiles = assemble_tiled_scalar_field(
            rho_tiles,
            parameter_set,
            parameter_set["tile_shape"],
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        _, rho_reference = self._deposit_and_assemble(particles, self._build_parameter_values(shape_factor=2), self._one_tile_parameters(parameter_set), dynamic_values)

        self.assertTrue(
            jnp.allclose(
                rho_from_tiles[1:-1, 1:-1, 1:-1],
                rho_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_public_compute_rho_rejects_flat_particles(self):
        parameter_set = self._build_parameter_values(shape_factor=2)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, self._one_tile_parameters(parameter_set))
        dynamic_values = {"alpha": 1.0}

        with self.assertRaisesRegex(TypeError, "non-array argument|abstract array"):
            compute_rho(
                self._particles(parameter_set),
                None,
                self._empty_scalar(parameter_set),
                *kernel_parameters_from_values(parameter_set, dynamic_values),
            )


if __name__ == "__main__":
    unittest.main()
