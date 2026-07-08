import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions import ghost_cells

jax.config.update("jax_enable_x64", True)


class TestGhostCells(unittest.TestCase):
    """Tests for the tiled ghost-cell boundary condition approach."""

    def setUp(self):
        self.tile_shape = (2, 2, 2)
        self.g = 1
        self.world = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }

    def test_update_tiled_ghost_cells_periodic_refreshes_neighbor_halos(self):
        # this tests the communication of ghost cells between two tiles in a periodic domain

        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        field_tiles = field_tiles.at[0, 0, 0, 1:3, 1:3, 1:3].set(1.0)
        field_tiles = field_tiles.at[1, 0, 0, 1:3, 1:3, 1:3].set(2.0)
        # create two tiles, one with a value of 1.0 and one with a value of 2.0 constant 
        # across the tile

        result = ghost_cells.update_tiled_ghost_cells(field_tiles, self.world, self.g, self.tile_shape)
        # call the update ghost cells method to communicate the ghost cells between the two tiles

        self.assertTrue(jnp.all(result[0, 0, 0, -1, 1:3, 1:3] == 2.0))
        # make sure the ghost cell on the first tile has been updated from 1.0 to 2.0
        self.assertTrue(jnp.all(result[1, 0, 0, 0, 1:3, 1:3] == 1.0))
        # make sure the ghost cell on the second tile has been updated from 2.0 to 1.0

    def test_fold_tiled_ghost_cells_periodic_adds_to_owner_tile(self):
        # this tests the folding of ghost cells back to the owner tile in a periodic domain
        # this is used to confirm the current and charge deposition is correct when using ghost cells

        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        # create two tiles with a shape of (4, 4, 4) and a ghost cell width of 1
        field_tiles = field_tiles.at[0, 0, 0, -1, 2, 2].set(3.0)
        # set the ghost cell on the first tile at the right x boundary to a value of 3.0
        field_tiles = field_tiles.at[1, 0, 0, 0, 2, 2].set(5.0)
        # set the ghost cell on the second tile at the left x boundary to a value of 5.0

        result = ghost_cells.fold_tiled_ghost_cells(field_tiles, self.world, self.g, self.tile_shape)
        # call the fold ghost cells method to add the ghost cell values back to the owner tile

        self.assertAlmostEqual(float(result[1, 0, 0, 1, 2, 2]), 3.0)
        # make sure the ghost cell value of 3.0 on the first tile has been added to the owner tile
        self.assertAlmostEqual(float(result[0, 0, 0, -2, 2, 2]), 5.0)
        # make sure the ghost cell value of 5.0 on the second tile has been added to the owner tile
        self.assertEqual(float(result[0, 0, 0, -1, 2, 2]), 0.0)
        self.assertEqual(float(result[1, 0, 0, 0, 2, 2]), 0.0)
        # make sure the ghost cell values have been reset to 0.0 after folding

    def test_apply_tiled_conducting_bc_zeros_global_tangential_faces(self):
        # this tests the application of conducting boundary conditions to a tiled electric field in a periodic domain

        world = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING},
        }
        # create a world with conducting boundary conditions in all directions

        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))
        # create a tuple of three electric field components (Ex, Ey, Ez) with shape (1, 1, 1, 4, 4, 4) and all values set to 1.0

        Ex, Ey, Ez = ghost_cells.apply_tiled_conducting_bc(E, world, self.g)
        # call the apply tiled conducting boundary conditions method to zero out the tangential faces of the electric field components

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))
        # make sure the tangential faces of the electric field components have been zeroed out

    def test_apply_tiled_scalar_conducting_bc_periodic_noop(self):
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        # create a scalar field with shape (1, 1, 1, 4, 4, 4) and values from 0 to 63
        result = ghost_cells.apply_tiled_scalar_conducting_bc(field, self.world, self.g)
        # call the apply tiled scalar conducting boundary conditions method
        # since the boundary conditions are periodic, this should be a no-op and the result should be equal to the input field

        self.assertTrue(jnp.allclose(result, field))
        # confirm the field has not been modified since the boundary conditions are periodic and should not affect the field


if __name__ == '__main__':
    unittest.main()
