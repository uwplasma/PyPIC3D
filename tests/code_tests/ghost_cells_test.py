import unittest
from types import SimpleNamespace

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
        self.parameter_set = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }

    def _parameters_with_field_mesh(self, tile_grid_shape):
        parameter_set = dict(self.parameter_set)
        try:
            parameter_set["field_mesh"] = ghost_cells.make_field_mesh(tile_grid_shape)
        except ValueError as exc:
            self.skipTest(str(exc))
        return SimpleNamespace(
            tile_shape=tuple(int(width) for width in parameter_set["tile_shape"]),
            guard_cells=self.g,
            boundary_conditions=(
                int(parameter_set["boundary_conditions"]["x"]),
                int(parameter_set["boundary_conditions"]["y"]),
                int(parameter_set["boundary_conditions"]["z"]),
            ),
            field_mesh=parameter_set["field_mesh"],
        )

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
        # make sure the ghost cell on the second tile has been updated from 2.0 to 1.0

    def test_update_tiled_ghost_cells_requires_static_field_mesh(self):
        # tiled halo exchange should use the startup-owned field mesh, not infer
        # a device mesh from the current array shape.

        field_tiles = jnp.zeros((1, 1, 1, 4, 4, 4))

        incomplete_parameters = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

        with self.assertRaises(AttributeError):
            ghost_cells.update_tiled_ghost_cells(field_tiles, incomplete_parameters, self.g)

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
        # make sure the ghost cell values have been reset to 0.0 after folding

    def test_apply_tiled_zero_boundary_zeros_global_tangential_faces(self):
        # this tests the application of axis-wise conducting boundary conditions to a tiled electric field

        parameter_set = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        )
        # create a parameter_set with conducting boundary conditions in all directions

        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))
        # create a tuple of three electric field components (Ex, Ey, Ez) with shape (1, 1, 1, 4, 4, 4) and all values set to 1.0

        Ex, Ey, Ez = E
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=0, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=0, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=1, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=1, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=2, num_guard_cells=self.g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=2, num_guard_cells=self.g)
        # call the shared scalar zero boundary method for each tangential electric component

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))
        # make sure the tangential faces of the electric field components have been zeroed out

    def test_apply_tiled_zero_boundary_periodic_axis_keeps_refreshed_field(self):
        parameter_set = self._parameters_with_field_mesh((1, 1, 1))
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        # create a scalar field with shape (1, 1, 1, 4, 4, 4) and values from 0 to 63
        refreshed = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)
        result = ghost_cells.apply_tiled_zero_boundary(refreshed, parameter_set, axis=0, num_guard_cells=self.g)
        # periodic field boundaries do not zero the physical plane

        self.assertTrue(jnp.allclose(result, refreshed))
        # confirm the field has not been modified after an already consistent periodic halo refresh


if __name__ == '__main__':
    unittest.main()
