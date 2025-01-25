import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.boris import boris, modified_boris

jax.config.update("jax_enable_x64", True)

class TestBorisMethods(unittest.TestCase):

    def setUp(self):
        self.q = 1.0
        self.m = 1.0
        self.x = jnp.array([0.0])
        self.y = jnp.array([0.0])
        self.z = jnp.array([0.0])
        # single particle at the origin

        self.vx = jnp.array([1.0])
        self.vy = jnp.array([0.0])
        self.vz = jnp.array([0.0])
        # single particle moving in the x direction

        self.Ex = jnp.zeros( (5, 5, 5) )
        self.Ey = jnp.zeros( (5, 5, 5) )
        self.Ez = jnp.zeros( (5, 5, 5) )
        self.Bx = jnp.zeros( (5, 5, 5) )
        self.By = jnp.zeros( (5, 5, 5) )
        self.Bz = jnp.ones( (5, 5, 5) )
        # grid of 5x5x5 with a uniform magnetic field in the z direction

        self.grid = jnp.arange(-1/2, 1/2, (1/5)), jnp.arange(-1/2, 1/2, (1/5)), jnp.arange(-1/2, 1/2, (1/5))
        self.staggered_grid = jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5)), jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5)), jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5))
        self.dt = 0.1
        # grid and staggered grid for a 5x5x5 grid with a spacing of 1/5 and a timestep of 0.1

    def test_boris(self):

        Ex_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ex, fill_value=0)
        Ey_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ey, fill_value=0)
        Ez_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ez, fill_value=0)

        Bx_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.Bx, fill_value=0)
        By_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.By, fill_value=0)
        Bz_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.Bz, fill_value=0)

        E_interpolate = (Ex_interpolate, Ey_interpolate, Ez_interpolate)
        B_interpolate = (Bx_interpolate, By_interpolate, Bz_interpolate)
        # create interpolators for the electric and magnetic fields
        newvx, newvy, newvz = boris(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.q, self.m, E_interpolate, B_interpolate, self.grid, self.staggered_grid, self.dt)
        self.assertIsInstance(newvx, jnp.ndarray)
        self.assertIsInstance(newvy, jnp.ndarray)
        self.assertIsInstance(newvz, jnp.ndarray)
        # make sure the velocities are jax arrays
        jnp.allclose(newvx, 1.0)
        # make sure the x velocity is unchanged
        jnp.allclose(newvy, 0.0)
        # make sure the y velocity is unchanged
        jnp.allclose(newvz, 1.0)
        # make sure the z velocity is 1.0 from the magnetic field

    def test_modified_boris(self):
        
        Ex_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ex, fill_value=0)
        Ey_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ey, fill_value=0)
        Ez_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.grid, self.Ez, fill_value=0)

        Bx_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.Bx, fill_value=0)
        By_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.By, fill_value=0)
        Bz_interpolate = jax.scipy.interpolate.RegularGridInterpolator(self.staggered_grid, self.Bz, fill_value=0)

        E_interpolate = (Ex_interpolate, Ey_interpolate, Ez_interpolate)
        B_interpolate = (Bx_interpolate, By_interpolate, Bz_interpolate)
        # create interpolators for the electric and magnetic fields

        w = jnp.zeros( (3,3) )
        g = jnp.zeros( (3,3) )
        newvx, newvy, newvz = modified_boris(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.q, self.m, E_interpolate, B_interpolate, w, g, self.grid, self.staggered_grid, self.dt)
        self.assertIsInstance(newvx, jnp.ndarray)
        self.assertIsInstance(newvy, jnp.ndarray)
        self.assertIsInstance(newvz, jnp.ndarray)
        # make sure the velocities are jax arrays
        jnp.allclose(newvx, 1.0)
        # make sure the x velocity is unchanged
        jnp.allclose(newvy, 0.0)
        # make sure the y velocity is unchanged
        jnp.allclose(newvz, 1.0)
        # make sure the z velocity is 1.0 from the magnetic field


if __name__ == '__main__':
    unittest.main()