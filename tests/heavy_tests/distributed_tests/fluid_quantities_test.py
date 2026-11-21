"""Focused cross-device regression tests."""

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
    def test_tile_major_velocity_runs_on_multi_tile_kernel_storage_when_devices_are_available(self):
        tile_shape = (4, 3, 2)
        n_tiles = (8 // tile_shape[0]) * (6 // tile_shape[1]) * (4 // tile_shape[2])
        if len(jax.devices()) < n_tiles:
            raise RuntimeError("multi-tile field mesh needs one logical device per tile")

        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=2, tile_shape=tile_shape)
        particles = self._spread_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        self.assertEqual(
            velocity_tiles.shape,
            (
                2,
                2,
                2,
                tile_shape[0] + 2 * int(static_parameters.guard_cells),
                tile_shape[1] + 2 * int(static_parameters.guard_cells),
                tile_shape[2] + 2 * int(static_parameters.guard_cells),
            ),
        )
        self.assertTrue(jnp.any(jnp.abs(velocity_tiles) > 0.0))
        self.assertFalse(jnp.any(jnp.isnan(velocity_tiles)))


    def test_tile_edge_deposits_reach_adjacent_tile_for_cic_and_tsc(self):
        if len(jax.devices()) < 2:
            raise RuntimeError("two-tile fluid velocity test needs two logical devices")

        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                x1=jnp.array([-0.25, 0.25]),
                x2=jnp.array([0.0, 0.0]),
                x3=jnp.array([0.0, 0.0]),
                v1=jnp.array([2.0, 10.0]),
            )
        ]

        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                tiled_static, tiled_dynamic = self._build_parameters(
                    shape_factor=shape_factor,
                    tile_shape=(4, 6, 4),
                )

                tiled_particles, tiled_species = build_tiled_particles(
                    particles,
                    tiled_static,
                    tiled_dynamic,
                )

                tiled_velocity = fluid_velocity(
                    tiled_particles,
                    tiled_species,
                    self._scalar_tiles(tiled_static, tiled_dynamic),
                    0,
                    tiled_static,
                    tiled_dynamic,
                )

                tiled_velocity = self._assemble_scalar(tiled_velocity, tiled_static)

                x_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[0])))
                y_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[1])))
                z_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[2])))
                self.assertAlmostEqual(float(tiled_velocity[x_index, y_index, z_index]), 6.0)


if __name__ == "__main__":
    unittest.main()
