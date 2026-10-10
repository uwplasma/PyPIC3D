"""Single-device numerical tests."""

from tests.support.fluid_quantities_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    BC_TYPE_PARTICLE,
    TiledFluidQuantitiesFixtures,
    build_field_output_map,
    build_tiled_particles,
    compute_rho,
    compute_velocity_field,
    fluid_velocity,
    fold_tiled_ghost_cells,
    jax,
    jnp,
    particle_species,
    particle_vector_reflecting_parity,
    unittest,
)


class TestTiledFluidQuantities(TiledFluidQuantitiesFixtures, unittest.TestCase):
    def test_fluid_velocity_computes_weighted_local_average(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))


    def test_compute_velocity_field_uses_selected_direction(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = compute_velocity_field(
            tiled_particles,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            1,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 2.0, rtol=1.0e-12, atol=1.0e-12))


    def test_field_output_map_adds_requested_particle_diagnostics(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(
            particles,
            static_parameters,
            dynamic_parameters,
        )

        scalar_field = self._scalar_tiles(static_parameters, dynamic_parameters)
        vector_field = (scalar_field, scalar_field, scalar_field)
        fields = (
            vector_field,
            vector_field,
            vector_field,
            scalar_field,
            scalar_field,
            (vector_field, vector_field),
            None,
            jnp.asarray(False),
        )

        field_map = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
        )
        self.assertEqual(tuple(field_map), ("E", "B", "J"))

        expected_rho = compute_rho(
            tiled_particles,
            species_config,
            scalar_field,
            static_parameters,
            dynamic_parameters,
        )

        field_map = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
            include_fluid_velocity=True,
            include_charge_density=True,
        )
        self.assertEqual(tuple(field_map), ("E", "B", "J", "rho", "fluid_velocity"))
        self.assertTrue(jnp.allclose(field_map["rho"], expected_rho))
        self.assertTrue(jnp.any(field_map["rho"] != 0.0))

        for velocity_component, expected_velocity in zip(
            field_map["fluid_velocity"],
            (8.0, 2.0, -1.0),
        ):
            occupied = jnp.abs(velocity_component) > 0.0
            self.assertTrue(jnp.any(occupied))
            self.assertTrue(
                jnp.allclose(
                    velocity_component[occupied],
                    expected_velocity,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )


    def test_inactive_slots_do_not_contribute_to_fluid_velocity(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(
            particles,
            static_parameters,
            dynamic_parameters,
            capacity_factor=2.0,
        )

        inactive = ~tiled_particles.active
        x = tiled_particles.x.at[inactive, 0].set(0.0)
        x = x.at[inactive, 1].set(0.0)
        x = x.at[inactive, 2].set(0.0)
        u = tiled_particles.u.at[inactive, 0].set(1000.0)
        noisy_tiled_particles = tiled_particles._replace(x=x, u=u)

        velocity_tiles = fluid_velocity(
            noisy_tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))


    def test_empty_cells_are_zero(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        self.assertTrue(jnp.any(velocity_tiles == 0.0))
        self.assertFalse(jnp.any(jnp.isnan(velocity_tiles)))




    def test_fluid_velocity_uses_particle_boundary_conditions(self):
        periodic_static, dynamic_parameters = self._build_parameters(shape_factor=1)
        field_conducting_static = periodic_static._replace(
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                x1=jnp.array([-1.75, 1.75]),
                x2=jnp.array([0.0, 0.0]),
                x3=jnp.array([0.0, 0.0]),
                v1=jnp.array([2.0, 10.0]),
            )
        ]
        periodic_particles, periodic_species = build_tiled_particles(
            particles,
            periodic_static,
            dynamic_parameters,
        )
        conducting_field_particles, conducting_field_species = build_tiled_particles(
            particles,
            field_conducting_static,
            dynamic_parameters,
        )

        periodic_velocity = fluid_velocity(
            periodic_particles,
            periodic_species,
            self._scalar_tiles(periodic_static, dynamic_parameters),
            0,
            periodic_static,
            dynamic_parameters,
        )
        conducting_field_velocity = fluid_velocity(
            conducting_field_particles,
            conducting_field_species,
            self._scalar_tiles(field_conducting_static, dynamic_parameters),
            0,
            field_conducting_static,
            dynamic_parameters,
        )

        self.assertTrue(
            jnp.allclose(
                conducting_field_velocity,
                periodic_velocity,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

        velocity = self._assemble_scalar(conducting_field_velocity, field_conducting_static)
        x_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[0] + 2.0)))
        y_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[1])))
        z_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[2])))
        self.assertAlmostEqual(float(velocity[x_index, y_index, z_index]), 6.0)


    def test_reflecting_wall_uses_even_weight_and_tangential_moment_parity(self):
        periodic_static, dynamic_parameters = self._build_parameters(shape_factor=2)
        reflecting_static = periodic_static._replace(
            particle_boundary_conditions=(
                BC_PERIODIC,
                BC_PERIODIC,
                BC_CONDUCTING,
            ),
        )
        g = int(reflecting_static.guard_cells)
        n = int(reflecting_static.tile_shape[2])
        deposited = self._scalar_tiles(reflecting_static, dynamic_parameters)

        # Moments sit on collocated nodes, and the upper wall node g+n is owned.
        # Number weight and a tangential moment add the nodal images (the wall
        # node receives its coincident image); a wall-normal moment subtracts
        # them and vanishes on the wall.
        self.assertEqual(g, 2)
        owned = (0, 0, 0, g, g, slice(g + n - 2, g + n + 1))
        deposited = deposited.at[owned].set(jnp.array([2.0, 3.0, 5.0]))
        deposited = deposited.at[0, 0, 0, g, g, g + n + 1].set(7.0)

        even_fold = fold_tiled_ghost_cells(
            deposited,
            reflecting_static,
            g,
            bc_type=BC_TYPE_PARTICLE,
        )
        tangential_fold = fold_tiled_ghost_cells(
            deposited,
            reflecting_static,
            g,
            bc_type=BC_TYPE_PARTICLE,
            reflecting_parity=particle_vector_reflecting_parity(0),
        )
        normal_fold = fold_tiled_ghost_cells(
            deposited,
            reflecting_static,
            g,
            bc_type=BC_TYPE_PARTICLE,
            reflecting_parity=particle_vector_reflecting_parity(2),
        )

        self.assertTrue(bool(jnp.allclose(even_fold[owned], jnp.array([2.0, 10.0, 10.0]))))
        self.assertTrue(bool(jnp.allclose(tangential_fold[owned], jnp.array([2.0, 10.0, 10.0]))))
        self.assertTrue(bool(jnp.allclose(normal_fold[owned], jnp.array([2.0, -4.0, 0.0]))))


    def test_reflecting_wall_fluid_velocity_remains_a_bounded_particle_average(self):
        periodic_static, dynamic_parameters = self._build_parameters(shape_factor=2)
        reflecting_static = periodic_static._replace(
            particle_boundary_conditions=(
                BC_PERIODIC,
                BC_PERIODIC,
                BC_CONDUCTING,
            ),
        )
        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-0.2, 0.2, -0.2, 0.2]),
                x2=jnp.zeros(4),
                x3=jnp.array([-0.99, -0.76, 0.76, 0.99]),
                v1=jnp.array([-0.4, 0.2, 0.1, 0.3]),
                v3=jnp.array([0.25, -0.15, 0.05, -0.2]),
            )
        ]
        tiled_particles, species_config = build_tiled_particles(
            particles,
            reflecting_static,
            dynamic_parameters,
        )
        g = int(reflecting_static.guard_cells)

        ux = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(reflecting_static, dynamic_parameters),
            0,
            reflecting_static,
            dynamic_parameters,
        )
        uz = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(reflecting_static, dynamic_parameters),
            2,
            reflecting_static,
            dynamic_parameters,
        )

        self.assertLessEqual(float(jnp.max(jnp.abs(ux[:, :, :, g:-g, g:-g, g:-g]))), 0.4 + 1.0e-12)
        self.assertLessEqual(float(jnp.max(jnp.abs(uz[:, :, :, g:-g, g:-g, g:-g]))), 0.25 + 1.0e-12)
        self.assertFalse(bool(jnp.any(jnp.isnan(ux))))
        self.assertFalse(bool(jnp.any(jnp.isnan(uz))))
        # nodal images about the lower wall node g: ghost g-k mirrors g+k
        for k in range(1, g + 1):
            self.assertTrue(bool(jnp.allclose(ux[0, 0, 0, g:-g, g:-g, g - k], ux[0, 0, 0, g:-g, g:-g, g + k])))
            self.assertTrue(bool(jnp.allclose(uz[0, 0, 0, g:-g, g:-g, g - k], -uz[0, 0, 0, g:-g, g:-g, g + k])))



if __name__ == "__main__":
    unittest.main()
