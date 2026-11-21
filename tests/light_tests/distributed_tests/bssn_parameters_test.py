from tests.support.bssn_parameters_runtime_fixtures import (
    bssn_to_tiles,
    jnp,
    runtime,
    tiles_to_bssn,
    unittest,
)


class BSSNParametersTest(unittest.TestCase):
    def test_component_leading_tile_roundtrip(self):
        static, *_ = runtime()
        static = static._replace(tile_shape=(4, 2, 1), field_mesh=None)
        values = jnp.arange(2 * 3 * 8 * 4).reshape(2, 3, 8, 4, 1)
        tiles = bssn_to_tiles(values, static)
        self.assertEqual(tiles.shape, (2, 3, 2, 2, 1, 10, 8, 7))
        self.assertTrue(bool(jnp.array_equal(tiles_to_bssn(tiles, static), values)))
        self.assertTrue(bool(jnp.array_equal(tiles[..., 0, 0, 0, 2, 3, 3], values[..., -1, 0, 0])))


if __name__ == "__main__":
    unittest.main()
