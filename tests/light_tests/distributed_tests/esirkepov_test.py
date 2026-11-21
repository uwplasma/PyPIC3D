from tests.support.distributed.esirkepov import TiledEsirkepovCurrentFixtures
"""Focused cross-device regression tests."""

from tests.support.esirkepov_fixtures import (
    BC_ABSORBING,
    BC_PERIODIC,
    Esirkepov_current,
    TiledParticles,
    assemble_tiled_vector_field,
    compute_rho,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    os,
    refresh_tiled_particle_tiles,
    tempfile,
    toml,
    unittest,
    update_tiled_particle_positions,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):
    def test_tiled_esirkepov_honors_one_guard_cell_startup_depth(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        parameter_set["guard_cells"] = 1
        x_old, u = self._basic_positions_and_velocities(parameter_set)

        self._assert_tiled_current_matches_reference(parameter_set, x_old, u, tile_shape=(2, 1, 1))


    def test_tiled_esirkepov_matches_global_1d_periodic_current(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(parameter_set, x_old, tile_shape)

        g = int(parameter_set["guard_cells"])
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        J_tiles = Esirkepov_current(
            tiled_particles,
            update_tiled_particle_positions(tiled_particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=tiled_particles.u,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=g)
        _, J_reference = self._assembled_esirkepov_current(
            parameter_set,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            dynamic_values,
            self._one_tile_shape_for_parameters(parameter_set),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )


    def test_tiled_esirkepov_uses_particle_boundaries_for_current_ghosts(self):
        tile_shape = (2, 1, 1)
        periodic_particle_parameters = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={"x": 0, "y": 0, "z": 0},
        )
        absorbing_particle_parameters = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={
                "x": BC_ABSORBING,
                "y": BC_PERIODIC,
                "z": BC_PERIODIC,
            },
        )
        periodic_particle_parameters["guard_cells"] = 2
        absorbing_particle_parameters["guard_cells"] = 2
        periodic_particle_parameters = self._parameters_with_tiled_grids(periodic_particle_parameters, tile_shape)
        absorbing_particle_parameters = self._parameters_with_tiled_grids(absorbing_particle_parameters, tile_shape)

        dx = periodic_particle_parameters["dx"]
        dt = periodic_particle_parameters["dt"]
        x_old = jnp.array([[1.75, 0.0, 0.0], [-1.75, 0.0, 0.0]])
        u = jnp.array([[0.6 * dx / dt, 0.0, 0.0], [-0.6 * dx / dt, 0.0, 0.0]])
        particles, species_config = self._particles_from_arrays(periodic_particle_parameters, tile_shape, x_old, u)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}

        _, _, J_template, _, _ = self._initialize_fields(periodic_particle_parameters, dynamic_values)
        static_periodic, dynamic_periodic = kernel_parameters_from_values(periodic_particle_parameters, dynamic_values)
        static_absorbing, dynamic_absorbing = kernel_parameters_from_values(absorbing_particle_parameters, dynamic_values)

        periodic_bc_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_periodic.dt),
            species_config,
            J_template,
            static_periodic,
            dynamic_periodic,
            coordinate_velocity=particles.u,
        )
        absorbing_bc_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_absorbing.dt),
            species_config,
            J_template,
            static_absorbing,
            dynamic_absorbing,
            coordinate_velocity=particles.u,
        )

        max_difference = max(
            float(jnp.max(jnp.abs(periodic_component - absorbing_component)))
            for periodic_component, absorbing_component in zip(periodic_bc_current, absorbing_bc_current)
        )
        self.assertGreater(max_difference, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
