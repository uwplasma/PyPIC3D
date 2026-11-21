from tests.support.distributed.yee import YeeTiledFixtures
"""Focused cross-device regression tests."""

from tests.support.yee_fixtures import (
    SimpleNamespace,
    assemble_tiled_vector_field,
    ghost_cells,
    build_tiled_yee_grids,
    build_yee_grid,
    jnp,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestYeeTiled(YeeTiledFixtures, unittest.TestCase):


    def test_tile_grid_axes_include_configured_guard_cells(self):
        parameter_set = self._build_parameter_values()
        parameter_set.update(x_min=1.25, y_min=-3.5, z_min=2.0, dx=0.25, dy=0.5, dz=0.75)
        tile_shape = (4, 3, 2)
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        dynamic_parameters = SimpleNamespace(
            dx=parameter_set["dx"],
            dy=parameter_set["dy"],
            dz=parameter_set["dz"],
            grids=SimpleNamespace(center=center_grid, vertex=vertex_grid),
        )
        mesh = ghost_cells.make_field_mesh((2, 2, 2))
        for field_mesh in (None, mesh):
            for g in (1, 2, 3):
                with self.subTest(sharded=field_mesh is not None, guard_cells=g):
                    static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g, field_mesh=field_mesh)
                    tiled_grids = build_tiled_yee_grids(static_parameters, dynamic_parameters)
                    for tiled_grid, stagger in zip(tiled_grids, (0.0, 0.5)):
                        for axis, name in enumerate("xyz"):
                            width = tile_shape[axis]
                            coordinates = tiled_grid[axis]
                            self.assertEqual(coordinates.shape, (2, 2, 2, width + 2 * g))
                            for tile in range(2):
                                expected = parameter_set[name + "_min"] + (
                                    tile * width + jnp.arange(-g, width + g) + stagger
                                ) * parameter_set["d" + name]
                                actual = jnp.take(coordinates, tile, axis=axis)
                                self.assertTrue(jnp.allclose(actual, expected, rtol=1.e-12, atol=1.e-12))


if __name__ == "__main__":
    unittest.main()
