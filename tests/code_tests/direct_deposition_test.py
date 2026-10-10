"""Single-device numerical tests."""

from tests.support.direct_deposition_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    DirectDepositionFixtures,
    J_from_rhov,
    get_first_order_weights,
    jnp,
    kernel_parameters_from_values,
    prepare_particle_axis_stencil,
    unittest,
)


class TestDirectDeposition(DirectDepositionFixtures, unittest.TestCase):
    def test_face_stencil_at_grid_node_is_independent_of_tile_origin(self):
        dx = 0.5
        position = jnp.asarray([-1.1102230246251565e-16])
        grid_axes = (
            jnp.asarray((-1.0, -0.5, 0.0, 0.5)),
            jnp.asarray((-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)),
        )
        face_stencils = []

        for grid_axis in grid_axes:
            face_grid_axis = grid_axis + 0.5 * dx
            _, _, delta_face, points_face = prepare_particle_axis_stencil(
                position,
                face_grid_axis,
                len(face_grid_axis),
                shape_factor=1,
                bc=BC_CONSTANT,
                ghost_cells=True,
            )
            weights_face, _, _ = get_first_order_weights(
                delta_face,
                delta_face,
                delta_face,
                dx,
                dx,
                dx,
            )
            weights_face = jnp.stack(weights_face)[:, 0]
            coordinates_face = face_grid_axis[0] + points_face[:, 0] * dx
            face_stencils.append((coordinates_face, weights_face))

            self.assertTrue(jnp.all(weights_face >= 0.0))
            self.assertAlmostEqual(float(jnp.sum(weights_face)), 1.0)

        for tiled_values, one_tile_values in zip(face_stencils[0], face_stencils[1]):
            self.assertTrue(jnp.allclose(tiled_values, one_tile_values))
        self.assertTrue(jnp.allclose(face_stencils[0][1], jnp.asarray((0.0, 0.5, 0.5))))


    def test_tiled_direct_deposition_masks_current_per_species_direction(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=4, Nz=4)
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 4,
            "particle_tile_nz": 4,
        }
        parameter_set = self._parameters_with_tiled_grids(
            parameter_set,
            self._tile_shape(simulation_parameters),
        )
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=2,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-0.75, -0.25, 0.25), (0.2, 0.3, 0.4), True),
                ((0, 0, 0), 1, 0, (0.75, 0.25, -0.25), (-0.5, -0.6, -0.7), True),
            ],
        )
        species_config = self._species_config(
            charges=[1.0, 2.0],
            masses=[1.0, 1.0],
            weights=[1.0, 1.0],
            update_x=[
                (False, True, False),
                (True, False, True),
            ],
        )
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        static_parameters = static_parameters._replace(current_filter="none")

        masked_current = J_from_rhov(
            particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )

        slot_mask = species_config.update_x.reshape((1, 1, 1, 2, 1, 3))
        reference_particles = particles._replace(u=jnp.where(slot_mask, particles.u, 0.0))
        reference_config = species_config._replace(update_x=jnp.ones_like(species_config.update_x))
        reference_current = J_from_rhov(
            reference_particles,
            reference_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )

        for masked_component, reference_component in zip(masked_current, reference_current):
            self.assertTrue(jnp.allclose(masked_component, reference_component))
            self.assertGreater(float(jnp.max(jnp.abs(masked_component))), 0.0)

        disabled_config = species_config._replace(update_x=jnp.zeros_like(species_config.update_x))
        disabled_current = J_from_rhov(
            particles,
            disabled_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        for component in disabled_current:
            self.assertTrue(jnp.allclose(component, 0.0))


    def test_tiled_direct_deposition_none_filter_does_not_use_alpha(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = self._one_tile_parameters(parameter_set)
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.1, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        _, raw_alpha_06 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 0.6},
            filter="none",
        )
        _, raw_alpha_10 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 1.0},
            filter="none",
        )
        _, digital_alpha_06 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 0.6},
            filter="digital",
        )

        for raw_06_component, raw_10_component in zip(raw_alpha_06, raw_alpha_10):
            self.assertTrue(jnp.allclose(raw_06_component, raw_10_component, rtol=1.0e-15, atol=1.0e-15))

        digital_difference = max(
            float(jnp.max(jnp.abs(raw_component - digital_component)))
            for raw_component, digital_component in zip(raw_alpha_06, digital_alpha_06)
        )
        self.assertGreater(digital_difference, 1.0e-12)


    def test_direct_current_refresh_uses_reflecting_vector_parity(self):
        parameter_set = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=4,
            dt=0.0,
            boundary_conditions={
                "x": BC_PERIODIC,
                "y": BC_PERIODIC,
                "z": BC_CONDUCTING,
            },
        )
        parameter_set["particle_boundary_conditions"] = {
            "x": BC_PERIODIC,
            "y": BC_PERIODIC,
            "z": BC_CONDUCTING,
        }
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 1,
            "particle_tile_nz": 4,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=2,
            slots=[
                ((0, 0, 0), 0, 0, (-0.2, 0.0, -0.99), (0.3, -0.2, 0.25), True),
                ((0, 0, 0), 0, 1, (0.2, 0.0, 0.99), (-0.1, 0.4, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])
        current_tiles, _ = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 1.0, "alpha": 1.0},
        )
        g = int(parameter_set["guard_cells"])
        n = simulation_parameters["particle_tile_nz"]

        # Jx and Jy are tangential to the z-walls and collocated in z: even
        # images about the wall nodes g and g+n, so ghost g-k mirrors g+k.
        for component in (0, 1):
            current = current_tiles[component][0, 0, 0]
            self.assertGreater(float(jnp.max(jnp.abs(current[g:-g, g:-g, g:-g]))), 0.0)
            for k in range(1, g + 1):
                self.assertTrue(jnp.allclose(current[g:-g, g:-g, g - k], current[g:-g, g:-g, g + k], atol=1.0e-12))
            for k in range(1, g):
                self.assertTrue(jnp.allclose(current[g:-g, g:-g, g + n + k], current[g:-g, g:-g, g + n - k], atol=1.0e-12))
        # Jz is normal and staggered in z: odd images about the wall faces
        current = current_tiles[2][0, 0, 0]
        self.assertGreater(float(jnp.max(jnp.abs(current[g:-g, g:-g, g:-g]))), 0.0)
        self.assertTrue(jnp.allclose(current[g:-g, g:-g, :g],
                                     -jnp.flip(current[g:-g, g:-g, g:2 * g], axis=-1), atol=1.0e-12))
        self.assertTrue(jnp.allclose(current[g:-g, g:-g, -g:],
                                     -jnp.flip(current[g:-g, g:-g, -2 * g:-g], axis=-1), atol=1.0e-12))



if __name__ == "__main__":
    unittest.main()
