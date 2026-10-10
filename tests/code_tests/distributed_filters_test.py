"""Single-device numerical tests."""

from tests.support.distributed_filters_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    DistributedFiltersFixtures,
    NamedSharding,
    _assemble_interior,
    _periodic_field,
    _static_parameters,
    _tile_interior,
    bilinear_filter,
    digital_filter,
    ghost_cells,
    jax,
    jnp,
    tiled_bilinear_filter,
    tiled_bilinear_filter_vector,
    tiled_digital_filter,
    unittest,
)


class TestDistributedFilters(DistributedFiltersFixtures, unittest.TestCase):
    def test_local_filters_read_guards_without_reading_another_tile_axis(self):
        g = 2
        tiles = jnp.zeros((2, 1, 1, 7, 7, 7), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, -g - 1, 3, 3].set(2.0)
        tiles = tiles.at[0, 0, 0, -g, 3, 3].set(12.0)
        changed_remote_interior = tiles.at[1, 0, 0, g, 3, 3].set(1000.0)

        digital = digital_filter(tiles, 0.6, num_guard_cells=g)
        changed_digital = digital_filter(changed_remote_interior, 0.6, num_guard_cells=g)
        bilinear = bilinear_filter(tiles, num_guard_cells=g)
        changed_bilinear = bilinear_filter(changed_remote_interior, num_guard_cells=g)

        self.assertEqual(digital[0, 0, 0, -g - 1, 3, 3], changed_digital[0, 0, 0, -g - 1, 3, 3])
        self.assertEqual(bilinear[0, 0, 0, -g - 1, 3, 3], changed_bilinear[0, 0, 0, -g - 1, 3, 3])
        self.assertNotEqual(digital[0, 0, 0, -g - 1, 3, 3], digital_filter(tiles.at[0, 0, 0, -g, 3, 3].set(0.0), 0.6, num_guard_cells=g)[0, 0, 0, -g - 1, 3, 3])
        self.assertNotEqual(bilinear[0, 0, 0, -g - 1, 3, 3], bilinear_filter(tiles.at[0, 0, 0, -g, 3, 3].set(0.0), num_guard_cells=g)[0, 0, 0, -g - 1, 3, 3])


    def test_tiled_filters_preserve_reduced_axis_behavior(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (6, 1, 1)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        interior = jnp.arange(6, dtype=jnp.float64).reshape((6, 1, 1))
        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)
        sharded_tiles = jax.device_put(tiles, sharding)

        digital = tiled_digital_filter(sharded_tiles, 0.6, static_parameters)
        bilinear = tiled_bilinear_filter(sharded_tiles, static_parameters)
        expected_digital = digital_filter(_periodic_field(interior, g), 0.6, num_guard_cells=g)
        expected_bilinear = bilinear_filter(_periodic_field(interior, g), num_guard_cells=g)

        self.assert_allclose(_assemble_interior(digital, g), expected_digital[g:-g, g:-g, g:-g])
        self.assert_allclose(_assemble_interior(bilinear, g), expected_bilinear[g:-g, g:-g, g:-g])


    def test_particle_filters_restore_reflecting_scalar_and_vector_halos(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (4, 1, 4)
        g = 2
        static_parameters = _static_parameters(
            mesh_shape, tile_shape, g, particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING))
        n = tile_shape[2]

        interior = jnp.arange(16, dtype=jnp.float64).reshape((4, 1, 4)) + 1.0
        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        scalar = tiled_digital_filter(
            tiles,
            0.6,
            static_parameters,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        # collocated scalars: nodal images about the wall nodes g and g+n
        for k in range(1, g + 1):
            self.assert_allclose(scalar[0, 0, 0, g:-g, g, g - k], scalar[0, 0, 0, g:-g, g, g + k])
        for k in range(1, g):
            self.assert_allclose(scalar[0, 0, 0, g:-g, g, g + n + k], scalar[0, 0, 0, g:-g, g, g + n - k])

        vector = tiled_bilinear_filter_vector(
            (tiles, 2.0 * tiles, 3.0 * tiles),
            static_parameters,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        # tangential components are collocated in z: even nodal images
        for component in (0, 1):
            current = vector[component][0, 0, 0, g:-g, g]
            for k in range(1, g + 1):
                self.assert_allclose(current[:, g - k], current[:, g + k])
            for k in range(1, g):
                self.assert_allclose(current[:, g + n + k], current[:, g + n - k])
        # the normal component is staggered in z: odd images about the wall faces
        current = vector[2][0, 0, 0, g:-g, g]
        self.assert_allclose(current[:, :g], -jnp.flip(current[:, g:2 * g], axis=-1))
        self.assert_allclose(current[:, -g:], -jnp.flip(current[:, -2 * g:-g], axis=-1))



if __name__ == "__main__":
    unittest.main()
