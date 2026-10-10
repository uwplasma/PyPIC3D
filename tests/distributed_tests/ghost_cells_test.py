"""Focused cross-device regression tests."""

from tests.support.ghost_cells_fixtures import (
    GhostCellsFixtures,
    ghost_cells,
    jnp,
    unittest,
)


class TestGhostCells(GhostCellsFixtures, unittest.TestCase):
    def test_update_tiled_ghost_cells_periodic_refreshes_neighbor_halos(self):
        # this tests the communication of ghost cells between two tiles in a periodic domain

        parameter_set = self._parameters_with_field_mesh((2, 1, 1))
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        field_tiles = field_tiles.at[0, 0, 0, 1:3, 1:3, 1:3].set(1.0)
        field_tiles = field_tiles.at[1, 0, 0, 1:3, 1:3, 1:3].set(2.0)
        # create two tiles, one with a value of 1.0 and one with a value of 2.0 constant 
        # across the tile

        result = ghost_cells.update_tiled_ghost_cells(field_tiles, parameter_set, self.g)
        # call the update ghost cells method to communicate the ghost cells between the two tiles

        self.assertTrue(jnp.all(result[0, 0, 0, -1, 1:3, 1:3] == 2.0))
        # make sure the ghost cell on the first tile has been updated from 1.0 to 2.0
        self.assertTrue(jnp.all(result[1, 0, 0, 0, 1:3, 1:3] == 1.0))


    def test_fold_tiled_ghost_cells_periodic_adds_to_owner_tile(self):
        # this tests the folding of ghost cells back to the owner tile in a periodic domain
        # this is used to confirm the current and charge deposition is correct when using ghost cells

        parameter_set = self._parameters_with_field_mesh((2, 1, 1))
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        # create two tiles with a shape of (4, 4, 4) and a ghost cell width of 1
        field_tiles = field_tiles.at[0, 0, 0, -1, 2, 2].set(3.0)
        # set the ghost cell on the first tile at the right x boundary to a value of 3.0
        field_tiles = field_tiles.at[1, 0, 0, 0, 2, 2].set(5.0)
        # set the ghost cell on the second tile at the left x boundary to a value of 5.0

        result = ghost_cells.fold_tiled_ghost_cells(field_tiles, parameter_set, self.g)
        # call the fold ghost cells method to add the ghost cell values back to the owner tile

        self.assertAlmostEqual(float(result[1, 0, 0, 1, 2, 2]), 3.0)
        # make sure the ghost cell value of 3.0 on the first tile has been added to the owner tile
        self.assertAlmostEqual(float(result[0, 0, 0, -2, 2, 2]), 5.0)
        # make sure the ghost cell value of 5.0 on the second tile has been added to the owner tile
        self.assertEqual(float(result[0, 0, 0, -1, 2, 2]), 0.0)
        self.assertEqual(float(result[1, 0, 0, 0, 2, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
