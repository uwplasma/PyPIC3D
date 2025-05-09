import unittest
import jax
import jax.numpy as jnp
import sys
import os


from PyPIC3D.boris import boris_single_particle
from PyPIC3D.utils import create_trilinear_interpolator

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

        Ex_interpolate = create_trilinear_interpolator(self.Ex, self.grid)
        Ey_interpolate = create_trilinear_interpolator(self.Ey, self.grid)
        Ez_interpolate = create_trilinear_interpolator(self.Ez, self.grid)
        Bx_interpolate = create_trilinear_interpolator(self.Bx, self.staggered_grid)
        By_interpolate = create_trilinear_interpolator(self.By, self.staggered_grid)
        Bz_interpolate = create_trilinear_interpolator(self.Bz, self.staggered_grid)
        # create interpolators for the electric and magnetic fields

        efield_atx = Ex_interpolate(self.x, self.y, self.z)
        efield_aty = Ey_interpolate(self.x, self.y, self.z)
        efield_atz = Ez_interpolate(self.x, self.y, self.z)
        # calculate the electric field at the particle positions
        bfield_atx = Bx_interpolate(self.x, self.y, self.z)
        bfield_aty = By_interpolate(self.x, self.y, self.z)
        bfield_atz = Bz_interpolate(self.x, self.y, self.z)

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



    def test_boris_single_particle(self):
        vx, vy, vz = 1.0, 0.0, 0.0
        x, y, z = 0.0, 0.0, 0.0

        q = 1.0
        m = 1.0
        E = jnp.array([0.0, 0.0, 0.0])
        B = jnp.array([0.0, 1.0, 0.0])
        dt = 0.1
        n_steps = 5000

        xs = []
        ys = []
        zs = []

        for i in range(n_steps):
            vx, vy, vz = boris_single_particle(vx, vy, vz, E[0], E[1], E[2], B[0], B[1], B[2], q, m, dt)
            x += vx * dt
            y += vy * dt
            z += vz * dt

            xs.append(x)
            ys.append(y)
            zs.append(z)


        def measure_xz_radius(xs, zs):
            """
            Measure the radius of the XZ cut by calculating the distance of each point
            in the XZ plane from the origin and returning the average and maximum radius.

            Parameters:
                xs (list): List of x-coordinates.
                zs (list): List of z-coordinates.

            Returns:
                tuple: A tuple containing the average radius and maximum radius.
            """
            xs = jnp.array(xs)
            zs = jnp.array(zs)
            # Calculate the distance of each point from the origin in the XZ plane
            radii = [jnp.sqrt(x**2 + z**2) for x, z in zip(xs, zs)]

            # Compute the average and maximum radius
            avg_radius = jnp.mean(jnp.asarray(radii))

            return avg_radius
        
        avg_radius  = measure_xz_radius(xs, zs)
        #print(f"Average radius: {avg_radius}")
        assert jnp.isclose(avg_radius, 1.28, atol=0.5)


if __name__ == '__main__':
    unittest.main()