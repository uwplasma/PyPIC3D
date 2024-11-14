import unittest
import numpy as np
from PyPIC3D.fdtd import periodic_laplacian, neumann_laplacian, dirichlet_laplacian, curlx, curly, curlz, update_B, update_E

import jax.numpy as jnp

class TestFDTDMethods(unittest.TestCase):

    def setUp(self):
        self.field = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.dt = 0.1
        self.C = 1.0

    def test_periodic_laplacian(self):
        result = periodic_laplacian(self.field, self.dx, self.dy, self.dz)
        expected = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_neumann_laplacian(self):
        result = neumann_laplacian(self.field, self.dx, self.dy, self.dz)
        expected = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_dirichlet_laplacian(self):
        result = dirichlet_laplacian(self.field, self.dx, self.dy, self.dz)
        expected = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_curlx(self):
        yfield = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        zfield = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = curlx(yfield, zfield, self.dy, self.dz)
        expected = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_curly(self):
        xfield = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        zfield = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = curly(xfield, zfield, self.dx, self.dz)
        expected = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_curlz(self):
        yfield = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        xfield = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = curlz(yfield, xfield, self.dx, self.dy)
        expected = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_update_B(self):
        Bx, By, Bz = self.field, self.field, self.field
        Ex, Ey, Ez = self.field, self.field, self.field
        result_Bx, result_By, result_Bz = update_B(Bx, By, Bz, Ex, Ey, Ez, self.dx, self.dy, self.dz, self.dt)
        expected_Bx, expected_By, expected_Bz = self.field, self.field, self.field
        np.testing.assert_array_almost_equal(result_Bx, expected_Bx)
        np.testing.assert_array_almost_equal(result_By, expected_By)
        np.testing.assert_array_almost_equal(result_Bz, expected_Bz)

    def test_update_E(self):
        Ex, Ey, Ez = self.field, self.field, self.field
        Bx, By, Bz = self.field, self.field, self.field
        result_Ex, result_Ey, result_Ez = update_E(Ex, Ey, Ez, Bx, By, Bz, self.dx, self.dy, self.dz, self.dt, self.C)
        expected_Ex, expected_Ey, expected_Ez = self.field, self.field, self.field
        np.testing.assert_array_almost_equal(result_Ex, expected_Ex)
        np.testing.assert_array_almost_equal(result_Ey, expected_Ey)
        np.testing.assert_array_almost_equal(result_Ez, expected_Ez)

if __name__ == '__main__':
    unittest.main()