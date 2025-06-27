import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.fields import initialize_fields, solve_poisson, calculateE, update_E, update_B
from PyPIC3D.utils import build_yee_grid

from PyPIC3D.pstd import spectral_gradient
from PyPIC3D.fdtd import centered_finite_difference_gradient

jax.config.update("jax_enable_x64", True)


class TestFieldsMethods(unittest.TestCase):

    def setUp(self):
        self.world = {
        'Nx': 100,
        'Ny': 100,
        'Nz': 100,
            'dx': 0.01,
            'dy': 0.01,
            'dz': 0.01,
            'dt': 0.01,
            'x_wind': 1.0,
            'y_wind': 1.0,
            'z_wind': 1.0
        }
        # Use normalized units to avoid numerical issues
        self.constants = {
            'eps': 1.0,  # Normalized permittivity
            'mu': 1.0,
            'C': 1.0
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

        # Make coordinate grid consistent with world dx, dy, dz
        dx, dy, dz = self.world['dx'], self.world['dy'], self.world['dz']

        x = jnp.linspace(0, 1, Nx)
        y = jnp.linspace(0, 1, Ny)
        z = jnp.linspace(0, 1, Nz)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')

        # Create a simpler analytical solution: phi = sin(2πx/L) * sin(2πy/L) * sin(2πz/L)
        # This has zero boundary conditions (periodic compatible)
        self.real_phi = jnp.sin(2 * jnp.pi * self.X) * jnp.sin(2 * jnp.pi * self.Y) * jnp.sin(2 * jnp.pi * self.Z)

        # For this phi, the Laplacian is: ∇²φ = -(2π/L)² * (sin_x*sin_y*sin_z + sin_x*sin_y*sin_z + sin_x*sin_y*sin_z)
        # = -3*(2π/L)² * phi
        # So rho = -∇²φ = 3*(2π/L)² * phi (with eps=1)
        wave_number_squared = (2 * jnp.pi)**2  # Assuming Lx=Ly=Lz=1
        self.rho = 3 * wave_number_squared * self.real_phi

        # Analytical E field: E = -∇φ
        self.real_Ex = -2 * jnp.pi * jnp.cos(2 * jnp.pi * self.X) * jnp.sin(2 * jnp.pi * self.Y) * jnp.sin(2 * jnp.pi * self.Z)
        self.real_Ey = -2 * jnp.pi * jnp.sin(2 * jnp.pi * self.X) * jnp.cos(2 * jnp.pi * self.Y) * jnp.sin(2 * jnp.pi * self.Z)
        self.real_Ez = -2 * jnp.pi * jnp.sin(2 * jnp.pi * self.X) * jnp.sin(2 * jnp.pi * self.Y) * jnp.cos(2 * jnp.pi * self.Z)

    def test_initialize_fields(self):
        self.assertEqual(self.Ex.shape, (100, 100, 100))
        self.assertEqual(self.Ey.shape, (100, 100, 100))
        self.assertEqual(self.Ez.shape, (100, 100, 100))
        self.assertEqual(self.Bx.shape, (100, 100, 100))
        self.assertEqual(self.By.shape, (100, 100, 100))
        self.assertEqual(self.Bz.shape, (100, 100, 100))
        self.assertEqual(self.Jx.shape, (100, 100, 100))
        self.assertEqual(self.Jy.shape, (100, 100, 100))
        self.assertEqual(self.Jz.shape, (100, 100, 100))
        self.assertEqual(self.phi.shape, (100, 100, 100))
        self.assertEqual(self.rho.shape, (100, 100, 100))

    def test_solve_poisson(self):
        phi = solve_poisson(self.rho, self.constants, self.world, self.phi, solver='fdtd')
        self.assertEqual(phi.shape, (100, 100, 100))
        jnp.allclose(phi, self.real_phi, atol=1e-4)
        # test if the solution is close to the analytical solution with finite difference method
        phi = solve_poisson(self.rho, self.constants, self.world, self.phi, solver='spectral')
        self.assertEqual(phi.shape, (100, 100, 100))
        jnp.allclose(phi, self.real_phi, atol=1e-4)
        # test if the solution is close to the analytical solution with spectral method

    def test_calculateE(self):
        """Test calculateE function with analytical solutions"""

        # Test 2: Test calculateE with empty particles (knowing rho will be zeroed)
        particles = []

        # Test with both solvers
        for solver_name in ['fdtd', 'spectral']:
            E, phi, rho = calculateE(self.world, particles, self.constants, self.rho, self.phi, solver_name, 'periodic')
            Ex, Ey, Ez = E

            # Check shapes
            self.assertEqual(Ex.shape, (100, 100, 100))
            self.assertEqual(Ey.shape, (100, 100, 100))
            self.assertEqual(Ez.shape, (100, 100, 100))
            self.assertEqual(phi.shape, (100, 100, 100))
            self.assertEqual(rho.shape, (100, 100, 100))

            # With empty particles, rho should be zero, so phi should also be zero
            print(f"Max phi value with no charge ({solver_name}): {jnp.max(jnp.abs(phi))}")
            print(f"Max E field value with no charge ({solver_name}): {jnp.max(jnp.abs(Ex))}")

            # Check that with zero charge density, phi and E are small
            self.assertLess(jnp.max(jnp.abs(phi)), 1e-10, f"Phi should be near zero with no charge ({solver_name})")
            self.assertLess(jnp.max(jnp.abs(Ex)), 1e-10, f"Ex should be near zero with no charge ({solver_name})")
            self.assertLess(jnp.max(jnp.abs(Ey)), 1e-10, f"Ey should be near zero with no charge ({solver_name})")
            self.assertLess(jnp.max(jnp.abs(Ez)), 1e-10, f"Ez should be near zero with no charge ({solver_name})")


        # Test gradient calculation with analytical phi
        Ex_grad_fdtd, Ey_grad_fdtd, Ez_grad_fdtd = centered_finite_difference_gradient(
            -1*self.real_phi, self.world['dx'], self.world['dy'], self.world['dz'], 'periodic')

        Ex_grad_spectral, Ey_grad_spectral, Ez_grad_spectral = spectral_gradient(-1*self.real_phi, self.world)

        jnp.allclose(Ex_grad_fdtd, self.real_Ex, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ey_grad_fdtd, self.real_Ey, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ez_grad_fdtd, self.real_Ez, rtol=1e-4, atol=1e-4)

        jnp.allclose(Ex_grad_spectral, self.real_Ex, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ey_grad_spectral, self.real_Ey, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ez_grad_spectral, self.real_Ez, rtol=1e-4, atol=1e-4)
        # ensure that the gradients match the analytical solution

        phi_computed = solve_poisson(self.rho, self.constants, self.world, self.phi, solver='spectral')
        # Use the computed phi from Poisson solver to get E field
        Ex_from_phi, Ey_from_phi, Ez_from_phi = spectral_gradient(-1*phi_computed, self.world)

        jnp.allclose(Ex_from_phi, self.real_Ex, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ey_from_phi, self.real_Ey, rtol=1e-4, atol=1e-4)
        jnp.allclose(Ez_from_phi, self.real_Ez, rtol=1e-4, atol=1e-4)
        # Ensure the computed E field matches the analytical solution

    def test_update_E(self):
        E = (self.Ex, self.Ey, self.Ez)
        B = (self.Bx, self.By, self.Bz)
        J = (self.Jx, self.Jy, self.Jz)
        Ex, Ey, Ez = update_E(E, B, J, self.world, self.constants, lambda x, y, z: (x, y, z))
        # Update the electric field using the update_E function
        self.assertEqual(Ex.shape, (100, 100, 100))
        self.assertEqual(Ey.shape, (100, 100, 100))
        self.assertEqual(Ez.shape, (100, 100, 100))

    def test_update_B(self):
        E = (self.Ex, self.Ey, self.Ez)
        B = (self.Bx, self.By, self.Bz)
        Bx, By, Bz = update_B(E, B, self.world, self.constants, lambda x, y, z: (x, y, z))
        # Update the magnetic field using the update_B function
        self.assertEqual(Bx.shape, (100, 100, 100))
        self.assertEqual(By.shape, (100, 100, 100))
        self.assertEqual(Bz.shape, (100, 100, 100))

if __name__ == '__main__':
    unittest.main()