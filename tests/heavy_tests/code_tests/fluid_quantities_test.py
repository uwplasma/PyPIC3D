"""One-tile API contracts."""

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
    jnp,
    particle_species,
    particle_vector_reflecting_parity,
    unittest,
)


class TestTiledFluidQuantities(TiledFluidQuantitiesFixtures, unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
