"""One-tile API contracts."""

from tests.support.esirkepov_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    Esirkepov_current,
    SpeciesConfig,
    TiledEsirkepovCurrentFixtures,
    TiledParticles,
    assemble_tiled_vector_field,
    build_static_parameters,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    os,
    tempfile,
    tile_vector_field,
    unittest,
    update_E,
    update_tiled_particle_positions,
    vector_field_for_output,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):


    def test_output_adapter_uses_parameter_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        parameter_set["guard_cells"] = 2
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["tile_shape"] = tile_shape
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        g = int(parameter_set["guard_cells"])
        E_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[0, 0, 0, 4, 2, 2].set(7.0)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        fields = (E_tiles, B_tiles, (Jx, Jy, Jz), rho, phi, (E_tiles, B_tiles), None)

        static_parameters = build_static_parameters(parameter_set)
        output_current = vector_field_for_output(fields[2], static_parameters)

        self.assertEqual(output_current[0].shape, shape)
        self.assertEqual(float(output_current[0][3, 1, 1]), 7.0)
        self.assertEqual(fields[2][0].shape[-3:], (12, 5, 5))

    def test_update_E_reads_two_guard_current_interior(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=1, Nz=1, dt=0.25)
        dynamic_values = {"C": 1.0, "eps": 2.0}
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["guard_cells"] = 2
        g = int(parameter_set["guard_cells"])
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        E_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 2:-2, 2:-2, 2:-2].set(4.0)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E_after, pml_state = update_E(E_tiles, B_tiles, (Jx, Jy, Jz), static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertTrue(jnp.allclose(E_after[0][:, :, :, g:-g, g:-g, g:-g], -0.5))

    def test_tiled_esirkepov_masks_current_per_species_direction(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=4, Nz=4, dt=0.05)
        parameter_set["guard_cells"] = 2
        tile_shape = (4, 4, 4)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        x = jnp.asarray(
            [[[[[[-0.75, -0.25, 0.25]], [[0.75, 0.25, -0.25]]]]]]
        )
        u = jnp.asarray(
            [[[[[[0.2, 0.3, 0.4]], [[-0.5, -0.6, -0.7]]]]]]
        )
        particles = TiledParticles(
            x=x,
            u=u,
            active=jnp.ones(x.shape[:-1], dtype=bool),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([1.0, 2.0]),
            mass=jnp.asarray([1.0, 1.0]),
            weight=jnp.asarray([1.0, 1.0]),
            update_x=jnp.asarray([
                (False, True, False),
                (True, False, True),
            ]),
        )
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(
            parameter_set,
            dynamic_values,
        )

        masked_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=particles.u,
        )

        slot_mask = species_config.update_x.reshape((1, 1, 1, 2, 1, 3))
        reference_particles = particles._replace(u=jnp.where(slot_mask, particles.u, 0.0))
        reference_config = species_config._replace(update_x=jnp.ones_like(species_config.update_x))
        reference_current = Esirkepov_current(
            reference_particles,
            update_tiled_particle_positions(reference_particles, reference_config, dynamic_parameters.dt),
            reference_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=reference_particles.u,
        )

        for masked_component, reference_component in zip(masked_current, reference_current):
            self.assertTrue(jnp.allclose(masked_component, reference_component))
            self.assertGreater(float(jnp.max(jnp.abs(masked_component))), 0.0)


    def test_public_Esirkepov_current_dispatches_tiled_particles_to_tile_local_current(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        parameter_set["guard_cells"] = 2
        parameter_set = self._parameters_with_tiled_grids(parameter_set, (8, 1, 1))
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(parameter_set, x_old, parameter_set["tile_shape"])

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
        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
            self.assertEqual(tile_component.shape[:3], (1, 1, 1))
        self._assert_single_tile_continuity(
            parameter_set, tiled_particles.x[tiled_particles.active],
            tiled_particles.u[tiled_particles.active], current=J_tiles)


if __name__ == "__main__":
    unittest.main()
