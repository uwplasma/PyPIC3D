import unittest
import jax
import jax.numpy as jnp
import sys
import os


from PyPIC3D.initialization import initialize_fields
from PyPIC3D.solvers.first_order_yee import update_E, update_B
from PyPIC3D.utils import build_yee_grid

from PyPIC3D.solvers.fdtd import centered_finite_difference_curl

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
            'C': 1.0,
            'alpha': 1.0,  # Digital filter alpha value
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

    def test_update_E(self):
        """Test update_E against analytical electromagnetic wave solution"""
        # Create analytical electromagnetic wave solution
        # Use a simple plane wave: E = E0 * sin(kz - ωt), B = (E0/c) * sin(kz - ωt)
        # Propagating in z-direction, E in x-direction, B in y-direction

        dx, dy, dz = self.world['dx'], self.world['dy'], self.world['dz']
        dt = self.world['dt']
        c = self.constants['C']
        eps = self.constants['eps']
        x_bc, y_bc, z_bc = 'periodic', 'periodic', 'periodic'

        # Wave parameters
        wavelength = 4 * dz * 32  # 4 wavelengths across domain
        k = 2 * jnp.pi / wavelength  # wavenumber
        omega = k * c  # frequency (dispersion relation)
        E0 = 1.0  # amplitude

        # Current time step (t=0)
        t = 0.0

        # Create wave fields at t=0
        kz_phase = k * self.Z
        Ex_initial = E0 * jnp.sin(kz_phase - omega * t)
        Ey_initial = jnp.zeros_like(Ex_initial)
        Ez_initial = jnp.zeros_like(Ex_initial)

        By_initial = (E0 / c) * jnp.sin(kz_phase + k * dz/2 - omega * t)
        Bx_initial = jnp.zeros_like(By_initial)
        Bz_initial = jnp.zeros_like(By_initial)

        # Zero current density
        Jx = jnp.zeros_like(Ex_initial)
        Jy = jnp.zeros_like(Ex_initial)
        Jz = jnp.zeros_like(Ex_initial)

        # Analytical solution at t + dt
        Ex_analytical = E0 * jnp.sin(kz_phase - omega * (t + dt))
        Ey_analytical = jnp.zeros_like(Ex_analytical)
        Ez_analytical = jnp.zeros_like(Ex_analytical)

        # Use curl function
        curl_func = lambda Ex, Ey, Ez: None  # curl function already defined in update_E

        # Test update_E
        E_initial = (Ex_initial, Ey_initial, Ez_initial)
        B_initial = (Bx_initial, By_initial, Bz_initial)
        J = (Jx, Jy, Jz)

        Ex_computed, Ey_computed, Ez_computed = update_E(E_initial, B_initial, J, self.world, self.constants, curl_func, x_bc, y_bc, z_bc)

        # Check shapes
        self.assertEqual(Ex_computed.shape, (100, 100, 100))
        self.assertEqual(Ey_computed.shape, (100, 100, 100))
        self.assertEqual(Ez_computed.shape, (100, 100, 100))

        # Compare with analytical solution (interior points to avoid boundary effects)
        interior = (slice(4, 28), slice(4, 28), slice(4, 28))

        Ex_error = jnp.max(jnp.abs(Ex_computed[interior] - Ex_analytical[interior]))
        Ey_error = jnp.max(jnp.abs(Ey_computed[interior] - Ey_analytical[interior]))
        Ez_error = jnp.max(jnp.abs(Ez_computed[interior] - Ez_analytical[interior]))

        # For a plane wave, Ey and Ez should remain zero
        self.assertLess(Ey_error, 1e-10, "Ey should remain zero for plane wave")
        self.assertLess(Ez_error, 1e-10, "Ez should remain zero for plane wave")

        # Ex should evolve correctly (tolerance accounts for discretization)
        relative_error = Ex_error / E0
        self.assertLess(relative_error, 5e-3, "Ex evolution should match analytical solution within 0.5%")

    def test_update_B(self):
        """Test update_B against analytical electromagnetic wave solution"""

        dx, dy, dz = self.world['dx'], self.world['dy'], self.world['dz']
        dt = self.world['dt']
        c = self.constants['C']
        x_bc, y_bc, z_bc = 'periodic', 'periodic', 'periodic'


        # Wave parameters (same as test_update_E)
        wavelength = 4 * dz * 32
        k = 2 * jnp.pi / wavelength
        omega = k * c
        E0 = 1.0

        # Current time step (t=0)
        t = 0.0

        # Create wave fields at t=0
        kz_phase = k * self.Z
        Ex_initial = E0 * jnp.sin(kz_phase - omega * t)
        Ey_initial = jnp.zeros_like(Ex_initial)
        Ez_initial = jnp.zeros_like(Ex_initial)

        By_initial = (E0 / c) * jnp.sin(kz_phase + k * dz/2 - omega * t)
        Bx_initial = jnp.zeros_like(By_initial)
        Bz_initial = jnp.zeros_like(By_initial)

        # Analytical solution at t + dt
        By_analytical = (E0 / c) * jnp.sin(kz_phase + k * dz/2 - omega * (t + dt))
        Bx_analytical = jnp.zeros_like(By_analytical)
        Bz_analytical = jnp.zeros_like(By_analytical)

        # Use curl function
        curl_func = lambda Ex, Ey, Ez: None  # curl function already defined in update_B

        # Test update_B
        E_initial = (Ex_initial, Ey_initial, Ez_initial)
        B_initial = (Bx_initial, By_initial, Bz_initial)

        Bx_computed, By_computed, Bz_computed = update_B(E_initial, B_initial, self.world, self.constants, curl_func, x_bc, y_bc, z_bc)

        # Check shapes
        self.assertEqual(Bx_computed.shape, (100, 100, 100))
        self.assertEqual(By_computed.shape, (100, 100, 100))
        self.assertEqual(Bz_computed.shape, (100, 100, 100))

        # Compare with analytical solution (interior points)
        interior = (slice(4, 28), slice(4, 28), slice(4, 28))

        Bx_error = jnp.max(jnp.abs(Bx_computed[interior] - Bx_analytical[interior]))
        By_error = jnp.max(jnp.abs(By_computed[interior] - By_analytical[interior]))
        Bz_error = jnp.max(jnp.abs(Bz_computed[interior] - Bz_analytical[interior]))

        # For a plane wave, Bx and Bz should remain zero
        self.assertLess(Bx_error, 1e-10, "Bx should remain zero for plane wave")
        self.assertLess(Bz_error, 1e-10, "Bz should remain zero for plane wave")

        # By should evolve correctly
        B0 = E0 / c
        relative_error = By_error / B0
        self.assertLess(relative_error, 5e-3, "By evolution should match analytical solution within 0.5%")

if __name__ == '__main__':
    unittest.main()