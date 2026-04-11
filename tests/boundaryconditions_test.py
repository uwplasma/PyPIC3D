import unittest
import jax
import jax.numpy as jnp
import sys
import os

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.boundaryconditions import (
    apply_supergaussian_boundary_condition, apply_zero_boundary_condition,
    update_ghost_cells, fold_ghost_cells, apply_conducting_bc,
    apply_scalar_conducting_bc
)

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
    """Tests for ghost cell functions used in the ghost cell boundary condition approach."""

    def setUp(self):
        # Small field with ghost cells: physical interior is 4x4x4, total 6x6x6
        self.Nx = 4
        self.Ny = 4
        self.Nz = 4
        key = jax.random.PRNGKey(42)
        self.field = jax.random.uniform(key, shape=(self.Nx + 2, self.Ny + 2, self.Nz + 2))
        # set ghost cells to known values for testing
        self.field = self.field.at[0, :, :].set(0.0)
        self.field = self.field.at[-1, :, :].set(0.0)
        self.field = self.field.at[:, 0, :].set(0.0)
        self.field = self.field.at[:, -1, :].set(0.0)
        self.field = self.field.at[:, :, 0].set(0.0)
        self.field = self.field.at[:, :, -1].set(0.0)
        # ghost cells are zero, interior has random data

    def test_update_ghost_cells_periodic(self):
        """Periodic BCs: ghost cells should copy from opposite interior edge."""
        result = update_ghost_cells(self.field, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)

        # x-axis: ghost[0] = interior[-2] (rightmost interior), ghost[-1] = interior[1] (leftmost interior)
        self.assertTrue(jnp.allclose(result[0, :, :], result[-2, :, :]))
        self.assertTrue(jnp.allclose(result[-1, :, :], result[1, :, :]))

        # y-axis: ghost[:,0,:] = interior[:,-2,:], ghost[:,-1,:] = interior[:,1,:]
        self.assertTrue(jnp.allclose(result[:, 0, :], result[:, -2, :]))
        self.assertTrue(jnp.allclose(result[:, -1, :], result[:, 1, :]))

        # z-axis: ghost[:,:,0] = interior[:,:,-2], ghost[:,:,-1] = interior[:,:,1]
        self.assertTrue(jnp.allclose(result[:, :, 0], result[:, :, -2]))
        self.assertTrue(jnp.allclose(result[:, :, -1], result[:, :, 1]))

    def test_update_ghost_cells_conducting(self):
        """Conducting BCs: ghost cells should be zero."""
        # put non-zero values in ghost cells first
        field = self.field.at[0, :, :].set(999.0).at[-1, :, :].set(999.0)
        field = field.at[:, 0, :].set(999.0).at[:, -1, :].set(999.0)
        field = field.at[:, :, 0].set(999.0).at[:, :, -1].set(999.0)

        result = update_ghost_cells(field, BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING)

        self.assertTrue(jnp.all(result[0, :, :] == 0.0))
        self.assertTrue(jnp.all(result[-1, :, :] == 0.0))
        self.assertTrue(jnp.all(result[:, 0, :] == 0.0))
        self.assertTrue(jnp.all(result[:, -1, :] == 0.0))
        self.assertTrue(jnp.all(result[:, :, 0] == 0.0))
        self.assertTrue(jnp.all(result[:, :, -1] == 0.0))

    def test_update_ghost_cells_mixed(self):
        """Mixed BCs: periodic in x, conducting in y and z."""
        result = update_ghost_cells(self.field, BC_PERIODIC, BC_CONDUCTING, BC_CONDUCTING)

        # x-axis should be periodic
        self.assertTrue(jnp.allclose(result[0, :, :], result[-2, :, :]))
        self.assertTrue(jnp.allclose(result[-1, :, :], result[1, :, :]))

        # y and z ghost cells should be zero
        self.assertTrue(jnp.all(result[:, 0, :] == 0.0))
        self.assertTrue(jnp.all(result[:, -1, :] == 0.0))
        self.assertTrue(jnp.all(result[:, :, 0] == 0.0))
        self.assertTrue(jnp.all(result[:, :, -1] == 0.0))

    def test_update_ghost_cells_preserves_interior(self):
        """Ghost cell update should not modify interior values."""
        interior_before = self.field[1:-1, 1:-1, 1:-1].copy()
        result = update_ghost_cells(self.field, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        self.assertTrue(jnp.allclose(result[1:-1, 1:-1, 1:-1], interior_before))

    def test_fold_ghost_cells_periodic(self):
        """Periodic fold: ghost cell deposits should be added to opposite interior cells."""
        field = jnp.zeros((6, 6, 6))
        # put deposits in ghost cells
        field = field.at[0, 2, 2].set(1.0)   # left x ghost
        field = field.at[-1, 2, 2].set(2.0)  # right x ghost
        field = field.at[2, 0, 2].set(3.0)   # left y ghost
        field = field.at[2, -1, 2].set(4.0)  # right y ghost
        field = field.at[2, 2, 0].set(5.0)   # left z ghost
        field = field.at[2, 2, -1].set(6.0)  # right z ghost
        # put a value in interior
        field = field.at[1, 2, 2].set(10.0)
        field = field.at[-2, 2, 2].set(20.0)

        result = fold_ghost_cells(field, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)

        # ghost[-1] (2.0) wraps to interior[1] (10.0) -> 12.0
        self.assertAlmostEqual(float(result[1, 2, 2]), 12.0)
        # ghost[0] (1.0) wraps to interior[-2] (20.0) -> 21.0
        self.assertAlmostEqual(float(result[-2, 2, 2]), 21.0)
        # all ghost cells should be cleared after folding
        self.assertEqual(float(result[0, 2, 2]), 0.0)
        self.assertEqual(float(result[-1, 2, 2]), 0.0)
        self.assertEqual(float(result[2, 0, 2]), 0.0)
        self.assertEqual(float(result[2, -1, 2]), 0.0)
        self.assertEqual(float(result[2, 2, 0]), 0.0)
        self.assertEqual(float(result[2, 2, -1]), 0.0)

    def test_fold_ghost_cells_conducting(self):
        """Conducting fold: ghost cells should be reflected."""
        field = jnp.zeros((6, 6, 6))
        field = field.at[0, 2, 2].set(1.0)
        field = field.at[-1, 2, 2].set(2.0)
        field = field.at[1, 2, 2].set(10.0)
        field = field.at[-2, 2, 2].set(20.0)

        result = fold_ghost_cells(field, BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING)

        # ghost[0] (1.0) reflects to same-side interior[1] (10.0) with sign flip -> 9.0
        self.assertAlmostEqual(float(result[1, 2, 2]), 9.0)
        # ghost[-1] (2.0) reflects to same-side interior[-2] (20.0) with sign flip -> 18.0
        self.assertAlmostEqual(float(result[-2, 2, 2]), 18.0)
        # ghost cells should be reflected
        self.assertEqual(float(result[0, 2, 2]), 0.0)
        self.assertEqual(float(result[-1, 2, 2]), 0.0)

    def test_apply_conducting_bc(self):
        """Conducting BCs should zero tangential E at boundaries."""
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        Ex = jax.random.uniform(keys[0], shape=(6, 6, 6))
        Ey = jax.random.uniform(keys[1], shape=(6, 6, 6))
        Ez = jax.random.uniform(keys[2], shape=(6, 6, 6))

        Ex_out, Ey_out, Ez_out = apply_conducting_bc((Ex, Ey, Ez), BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING)

        # x-boundary conducting: Ey and Ez tangential components zeroed at x faces
        self.assertTrue(jnp.all(Ey_out[1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ey_out[-2, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez_out[1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez_out[-2, :, :] == 0.0))

        # y-boundary conducting: Ex and Ez tangential components zeroed at y faces
        self.assertTrue(jnp.all(Ex_out[:, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex_out[:, -2, :] == 0.0))
        self.assertTrue(jnp.all(Ez_out[:, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez_out[:, -2, :] == 0.0))

        # z-boundary conducting: Ex and Ey tangential components zeroed at z faces
        self.assertTrue(jnp.all(Ex_out[:, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ex_out[:, :, -2] == 0.0))
        self.assertTrue(jnp.all(Ey_out[:, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey_out[:, :, -2] == 0.0))

    def test_apply_conducting_bc_periodic_noop(self):
        """When all axes are periodic, apply_conducting_bc should not modify E."""
        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, 3)
        Ex = jax.random.uniform(keys[0], shape=(6, 6, 6))
        Ey = jax.random.uniform(keys[1], shape=(6, 6, 6))
        Ez = jax.random.uniform(keys[2], shape=(6, 6, 6))

        Ex_out, Ey_out, Ez_out = apply_conducting_bc((Ex, Ey, Ez), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)

        self.assertTrue(jnp.allclose(Ex_out, Ex))
        self.assertTrue(jnp.allclose(Ey_out, Ey))
        self.assertTrue(jnp.allclose(Ez_out, Ez))

    def test_apply_scalar_conducting_bc(self):
        """Scalar conducting BC should zero field at conducting boundary faces."""
        key = jax.random.PRNGKey(2)
        field = jax.random.uniform(key, shape=(6, 6, 6))

        result = apply_scalar_conducting_bc(field, BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING)

        # field at first and last interior faces should be zero
        self.assertTrue(jnp.all(result[1, :, :] == 0.0))
        self.assertTrue(jnp.all(result[-2, :, :] == 0.0))
        self.assertTrue(jnp.all(result[:, 1, :] == 0.0))
        self.assertTrue(jnp.all(result[:, -2, :] == 0.0))
        self.assertTrue(jnp.all(result[:, :, 1] == 0.0))
        self.assertTrue(jnp.all(result[:, :, -2] == 0.0))

    def test_apply_scalar_conducting_bc_periodic_noop(self):
        """When all axes are periodic, scalar conducting BC should not modify field."""
        key = jax.random.PRNGKey(3)
        field = jax.random.uniform(key, shape=(6, 6, 6))

        result = apply_scalar_conducting_bc(field, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)

        self.assertTrue(jnp.allclose(result, field))


if __name__ == '__main__':
    unittest.main()
