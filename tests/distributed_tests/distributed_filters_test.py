"""Focused cross-device regression tests."""

from tests.support.distributed_filters_fixtures import (
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
    def test_tiled_digital_filter_stays_distributed_and_matches_global_stencil(self):
        self._scalar_filter_case(tiled_digital_filter, digital_filter, alpha=0.6)


    def test_tiled_bilinear_filter_stays_distributed_and_matches_global_stencil(self):
        self._scalar_filter_case(tiled_bilinear_filter, bilinear_filter)


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


if __name__ == "__main__":
    unittest.main()
