import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.fields import initialize_fields, solve_poisson, calculateE, update_E, update_B
from PyPIC3D.utils import build_yee_grid

jax.config.update("jax_enable_x64", True)


class TestFieldsMethods(unittest.TestCase):

    def setUp(self):
        self.world = {
            'Nx': 10,
            'Ny': 10,
            'Nz': 10,
            'dx': 0.1,
            'dy': 0.1,
            'dz': 0.1,
            'dt': 0.01,
            'x_wind': 1.0,
            'y_wind': 1.0,
            'z_wind': 1.0
        }
        self.constants = {
            'eps': 8.854e-12,
            'mu': 1.2566370613e-6,
            'C': 3e8
        }
        self.grid, self.staggered_grid = build_yee_grid(self.world)
        Nx = self.world['Nx']
        Ny = self.world['Ny']
        Nz = self.world['Nz']

        E, B, J, phi, rho = initialize_fields(Nx, Ny, Nz)
        self.Ex, self.Ey, self.Ez = E
        self.Bx, self.By, self.Bz = B
        self.Jx, self.Jy, self.Jz = J
        self.phi = phi
        self.rho = rho
        # create a grid for the fields

        x = jnp.linspace(0, 1, 10)
        y = jnp.linspace(0, 1, 10)
        z = jnp.linspace(0, 1, 10)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        self.real_phi = self.X**2 + self.Y**2 + self.Z**2
        self.rho = -6 * jnp.ones_like(self.phi)
        # create analytical solution for phi and rho
        # self.real_Ex = -2 * self.X
        # self.real_Ey = -2 * self.Y
        # self.real_Ez = -2 * self.Z
        # # create analytical solution for E

    def test_initialize_fields(self):
        self.assertEqual(self.Ex.shape, (10, 10, 10))
        self.assertEqual(self.Ey.shape, (10, 10, 10))
        self.assertEqual(self.Ez.shape, (10, 10, 10))
        self.assertEqual(self.Bx.shape, (10, 10, 10))
        self.assertEqual(self.By.shape, (10, 10, 10))
        self.assertEqual(self.Bz.shape, (10, 10, 10))
        self.assertEqual(self.Jx.shape, (10, 10, 10))
        self.assertEqual(self.Jy.shape, (10, 10, 10))
        self.assertEqual(self.Jz.shape, (10, 10, 10))
        self.assertEqual(self.phi.shape, (10, 10, 10))
        self.assertEqual(self.rho.shape, (10, 10, 10))

    def test_solve_poisson(self):
        phi = solve_poisson(self.rho, self.constants, self.world, self.phi, solver='fdtd')
        self.assertEqual(phi.shape, (10, 10, 10))
        jnp.allclose(phi, self.real_phi, atol=1e-2)
        # test if the solution is close to the analytical solution with finite difference method
        phi = solve_poisson(self.rho, self.constants, self.world, self.phi, solver='spectral')
        self.assertEqual(phi.shape, (10, 10, 10))
        jnp.allclose(phi, self.real_phi, atol=1e-2)
        # test if the solution is close to the analytical solution with spectral method

    def test_calculateE(self):
        particles = []  # Add appropriate particle initialization
        E, phi, rho = calculateE(self.world, particles, self.constants, self.rho, self.phi, None, 'fdtd', 'periodic')
        Ex, Ey, Ez = E
        self.assertEqual(Ex.shape, (10, 10, 10))
        self.assertEqual(Ey.shape, (10, 10, 10))
        self.assertEqual(Ez.shape, (10, 10, 10))
        E, phi, rho = calculateE(self.world, particles, self.constants, self.rho, self.phi, None, 'spectral', 'periodic')
        Ex, Ey, Ez = E
        self.assertEqual(Ex.shape, (10, 10, 10))
        self.assertEqual(Ey.shape, (10, 10, 10))
        self.assertEqual(Ez.shape, (10, 10, 10))

    def test_update_E(self):
        E = (self.Ex, self.Ey, self.Ez)
        B = (self.Bx, self.By, self.Bz)
        J = (self.Jx, self.Jy, self.Jz)
        Ex, Ey, Ez = update_E(E, B, J, self.world, self.constants, lambda x, y, z: (x, y, z))
        # Update the electric field using the update_E function
        self.assertEqual(Ex.shape, (10, 10, 10))
        self.assertEqual(Ey.shape, (10, 10, 10))
        self.assertEqual(Ez.shape, (10, 10, 10))

    def test_update_B(self):
        E = (self.Ex, self.Ey, self.Ez)
        B = (self.Bx, self.By, self.Bz)
        Bx, By, Bz = update_B(E, B, self.world, self.constants, lambda x, y, z: (x, y, z))
        # Update the magnetic field using the update_B function
        self.assertEqual(Bx.shape, (10, 10, 10))
        self.assertEqual(By.shape, (10, 10, 10))
        self.assertEqual(Bz.shape, (10, 10, 10))

if __name__ == '__main__':
    unittest.main()