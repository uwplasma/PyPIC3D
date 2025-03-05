import unittest
import jax
import jax.numpy as jnp
import sys
import os


from PyPIC3D.boris import boris_single_particle

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

        points = jnp.stack([self.x, self.y, self.z], axis=-1)

        efield_atx = Ex_interpolate(points)
        efield_aty = Ey_interpolate(points)
        efield_atz = Ez_interpolate(points)

        bfield_atx = Bx_interpolate(points)
        bfield_aty = By_interpolate(points)
        bfield_atz = Bz_interpolate(points)

        boris_vmap = jax.vmap(boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None))
        newvx, newvy, newvz = boris_vmap(self.vx, self.vy, self.vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, self.q, self.m, self.dt)

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