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


if __name__ == "__main__":
    unittest.main()
