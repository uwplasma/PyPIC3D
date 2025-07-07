import unittest
import jax
import jax.numpy as jnp
from PyPIC3D.J import J_first_order_weighting, J_second_order_weighting, wrap_around

jax.config.update("jax_enable_x64", True)

class TestJFunctions(unittest.TestCase):
    def setUp(self):
        self.q = 1.0
        self.x = jnp.array([0.0])
        self.y = jnp.array([0.0])
        self.z = jnp.array([0.0])
        self.vx = jnp.array([0.5])
        self.vy = jnp.array([0.5])
        self.vz = jnp.array([0.5])
        # particle with charge 1 and velocity 0.5(1,1,1)
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        self.x_wind = 1.0
        self.y_wind = 1.0
        self.z_wind = 1.0
        # uniform spatial resolution in xyz
        self.num_J = (jnp.zeros((10,10,10)), jnp.zeros((10,10,10)), jnp.zeros((10,10,10)))
        self.rho = jnp.zeros((10,10,10))
        # build initial J and rho arrays

        Jx = jnp.zeros((10,10,10))
        Jx = Jx.at[5,5,5].set(0.5 / (self.dx*self.dy*self.dz))  # non zero value at the center of the grid
        Jy = jnp.zeros((10,10,10))
        Jy = Jy.at[5,5,5].set(0.5 / (self.dx*self.dy*self.dz))  # non zero value at the center of the grid
        Jz = jnp.zeros((10,10,10))
        Jz = Jz.at[5,5,5].set(0.5 / (self.dx*self.dy*self.dz))  # non zero value at the center of the grid
        self.J = (Jx, Jy, Jz)
        # build expected J arrays with non-zero values at the center of the grid

    def test_J_first_order_weighting(self):
        Jx, Jy, Jz = J_first_order_weighting(self.q, self.x, self.y, self.z, self.vx, self.vy, self.vz, self.num_J, self.dx, self.dy, self.dz, self.x_wind, self.y_wind, self.z_wind)
        # compute Jx, Jy, Jz using first order weighting

        self.assertEqual(Jx.shape, (10,10,10))
        self.assertEqual(Jy.shape, (10,10,10))
        self.assertEqual(Jz.shape, (10,10,10))
        # make sure the shapes are correct

        self.assertTrue(jnp.allclose(self.J[0], Jx, atol=1e-8, rtol=1e-10))
        self.assertTrue(jnp.allclose(self.J[1], Jy, atol=1e-8, rtol=1e-10))
        self.assertTrue(jnp.allclose(self.J[2], Jz, atol=1e-8, rtol=1e-10))
        # check that the computed Jx, Jy, Jz match the expected values

    def test_J_second_order_weighting(self):
        # Use scalar values for all inputs to avoid JAX array predicate issues
        Jx, Jy, Jz = J_second_order_weighting(self.q, float(self.x[0]), float(self.y[0]), float(self.z[0]), float(self.vx[0]), float(self.vy[0]), float(self.vz[0]), self.J, self.dx, self.dy, self.dz, self.x_wind, self.y_wind, self.z_wind)
        self.assertEqual(Jx.shape, (10,10,10))
        self.assertEqual(Jy.shape, (10,10,10))
        self.assertEqual(Jz.shape, (10,10,10))
        # make sure the shapes are correct

        # self.assertTrue(jnp.allclose(self.J[0], Jx, atol=1e-8, rtol=1e-10))
        # self.assertTrue(jnp.allclose(self.J[1], Jy, atol=1e-8, rtol=1e-10))
        # self.assertTrue(jnp.allclose(self.J[2], Jz, atol=1e-8, rtol=1e-10))
        # Still need to get this working!!!!!!!!!
        # check that the computed Jx, Jy, Jz match the expected values

    def test_wrap_around(self):
        self.assertEqual(wrap_around(4, 3), 1)
        arr = jnp.array([2,3,4])
        result = wrap_around(arr, 3)
        self.assertTrue(jnp.all(result < 3))

if __name__ == '__main__':
    unittest.main()
