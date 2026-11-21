"""Focused cross-device regression tests."""

from tests.support.distributed.distributed_filters import (
    BC_CONDUCTING,
    BC_PERIODIC,
    _assemble_interior,
    _periodic_field,
)
from tests.support.distributed.distributed_filters import (
    DistributedFiltersFixtures,
    NamedSharding,
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
    tiled_digital_filter_vector,
    unittest,
)


class TestDistributedFilters(DistributedFiltersFixtures, unittest.TestCase):


    def test_tiled_vector_filters_preserve_stacked_and_tuple_layouts(self):
        mesh_shape = (2, 2, 2)
        tile_shape = (3, 3, 3)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        interior = jnp.arange(6 * 6 * 6, dtype=jnp.float64).reshape((6, 6, 6)) / 19.0
        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        stacked = jnp.stack((tiles, -2.0 * tiles, tiles + 3.0), axis=0)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.VECTOR_TILE_SPEC)
        sharded_stacked = jax.device_put(stacked, sharding)

        stacked_digital = tiled_digital_filter_vector(sharded_stacked, 0.55, static_parameters)
        tuple_digital = tiled_digital_filter_vector(tuple(sharded_stacked[i] for i in range(3)), 0.55, static_parameters)
        stacked_bilinear = tiled_bilinear_filter_vector(sharded_stacked, static_parameters)
        tuple_bilinear = tiled_bilinear_filter_vector(tuple(sharded_stacked[i] for i in range(3)), static_parameters)

        self.assertEqual(stacked_digital.sharding, sharding)
        self.assertEqual(stacked_bilinear.sharding, sharding)
        self.assertIsInstance(tuple_digital, tuple)
        self.assertIsInstance(tuple_bilinear, tuple)
        for component in range(3):
            self.assert_allclose(tuple_digital[component], stacked_digital[component])
            self.assert_allclose(tuple_bilinear[component], stacked_bilinear[component])


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


if __name__ == "__main__":
    unittest.main()
