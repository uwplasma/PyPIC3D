import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.boundaryconditions import (
    apply_supergaussian_boundary_condition, apply_zero_boundary_condition,
)
from PyPIC3D.boundary_conditions import ghost_cells

jax.config.update("jax_enable_x64", True)

class TestBoundaryConditions(unittest.TestCase):

    def test_apply_zero_boundary_condition(self):
        key = jax.random.PRNGKey(0)
        field = jax.random.uniform(key, shape=(10, 10, 10))

        result = apply_zero_boundary_condition(field)

        # Check if the boundary values are set to zero
        self.assertTrue(jnp.all(result[0, :, :] == 0))
        self.assertTrue(jnp.all(result[-1, :, :] == 0))
        self.assertTrue(jnp.all(result[:, 0, :] == 0))
        self.assertTrue(jnp.all(result[:, -1, :] == 0))
        self.assertTrue(jnp.all(result[:, :, 0] == 0))
        self.assertTrue(jnp.all(result[:, :, -1] == 0))


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
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        field_tiles = field_tiles.at[0, 0, 0, 1:3, 1:3, 1:3].set(1.0)
        field_tiles = field_tiles.at[1, 0, 0, 1:3, 1:3, 1:3].set(2.0)

        result = ghost_cells.update_tiled_ghost_cells(field_tiles, self.world, self.g, self.tile_shape)

        self.assertTrue(jnp.all(result[0, 0, 0, -1, 1:3, 1:3] == 2.0))
        self.assertTrue(jnp.all(result[1, 0, 0, 0, 1:3, 1:3] == 1.0))

    def test_fold_tiled_ghost_cells_periodic_adds_to_owner_tile(self):
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        field_tiles = field_tiles.at[0, 0, 0, -1, 2, 2].set(3.0)
        field_tiles = field_tiles.at[1, 0, 0, 0, 2, 2].set(5.0)

        result = ghost_cells.fold_tiled_ghost_cells(field_tiles, self.world, self.g, self.tile_shape)

        self.assertAlmostEqual(float(result[1, 0, 0, 1, 2, 2]), 3.0)
        self.assertAlmostEqual(float(result[0, 0, 0, -2, 2, 2]), 5.0)
        self.assertEqual(float(result[0, 0, 0, -1, 2, 2]), 0.0)
        self.assertEqual(float(result[1, 0, 0, 0, 2, 2]), 0.0)

    def test_apply_tiled_conducting_bc_zeros_global_tangential_faces(self):
        world = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING},
        }
        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))

        Ex, Ey, Ez = ghost_cells.apply_tiled_conducting_bc(E, world, self.g)

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))

    def test_apply_tiled_scalar_conducting_bc_periodic_noop(self):
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))

        result = ghost_cells.apply_tiled_scalar_conducting_bc(field, self.world, self.g)

        self.assertTrue(jnp.allclose(result, field))


if __name__ == '__main__':
    unittest.main()
