import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.solvers.electrostatic_yee import (
    solve_poisson_with_fft,
    solve_poisson_with_conjugate_gradient,
    calculate_electrostatic_fields,
)
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient
from PyPIC3D.solvers.pstd import spectral_gradient
from PyPIC3D.utils import build_yee_grid

jax.config.update("jax_enable_x64", True)


def apply_negative_laplacian(field, dx, dy, dz):
    laplacian_x = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2.0 * field) / (dx * dx)
    laplacian_y = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2.0 * field) / (dy * dy)
    laplacian_z = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2.0 * field) / (dz * dz)
    return -(laplacian_x + laplacian_y + laplacian_z)


class TestElectrostaticYeeMethods(unittest.TestCase):
    def setUp(self):
        self.Nx = 16
        self.Ny = 16
        self.Nz = 16
        self.x_wind = 2 * jnp.pi
        self.y_wind = 2 * jnp.pi
        self.z_wind = 2 * jnp.pi
        self.dx = self.x_wind / self.Nx
        self.dy = self.y_wind / self.Ny
        self.dz = self.z_wind / self.Nz

        x = jnp.linspace(0, self.x_wind, self.Nx, endpoint=False)
        y = jnp.linspace(0, self.y_wind, self.Ny, endpoint=False)
        z = jnp.linspace(0, self.z_wind, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')

        self.constants = {
            "eps": 1.0,
            "alpha": 1.0,
        }
        self.world = {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nz": self.Nz,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "x_wind": self.x_wind,
            "y_wind": self.y_wind,
            "z_wind": self.z_wind,
        }
        vertex_grid, center_grid = build_yee_grid(self.world)
        self.world["grids"] = {
            "vertex": vertex_grid,
            "center": center_grid,
        }

        self.initial_rho = jnp.zeros((self.Nx, self.Ny, self.Nz))
        self.initial_phi = jnp.zeros((self.Nx, self.Ny, self.Nz))

        self.particles = [
            particle_species(
                name="test",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=1.0,
                v1=jnp.zeros(1),
                v2=jnp.zeros(1),
                v3=jnp.zeros(1),
                x1=jnp.array([0.1]),
                x2=jnp.array([0.2]),
                x3=jnp.array([0.3]),
                xwind=self.x_wind,
                ywind=self.y_wind,
                zwind=self.z_wind,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                weight=1.0,
                x_bc="periodic",
                y_bc="periodic",
                z_bc="periodic",
                shape=1,
                dt=0.0,
            )
        ]

    def test_solve_poisson_with_fft_single_mode(self):
        phi_true = jnp.sin(self.X + self.Y + self.Z)
        rho = 3.0 * self.constants["eps"] * phi_true

        phi_num = solve_poisson_with_fft(rho, self.constants, self.world)
        phi_num = phi_num - jnp.mean(phi_num)
        phi_true = phi_true - jnp.mean(phi_true)

        self.assertEqual(phi_num.shape, phi_true.shape)
        self.assertTrue(jnp.allclose(phi_num, phi_true, atol=1e-10, rtol=1e-8))

    def test_solve_poisson_with_conjugate_gradient_single_mode(self):
        phi_true = jnp.sin(self.X + self.Y + self.Z)
        rhs = apply_negative_laplacian(phi_true, self.dx, self.dy, self.dz)
        rho = rhs * self.constants["eps"]

        phi_num = solve_poisson_with_conjugate_gradient(
            rho,
            self.initial_phi,
            self.constants,
            self.world,
            tol=1e-10,
            max_iter=4000,
        )
        phi_num = phi_num - jnp.mean(phi_num)
        phi_true = phi_true - jnp.mean(phi_true)

        self.assertEqual(phi_num.shape, phi_true.shape)
        self.assertTrue(jnp.allclose(phi_num, phi_true, atol=1e-7, rtol=1e-6))

        residual = apply_negative_laplacian(phi_num, self.dx, self.dy, self.dz) - rho / self.constants["eps"]
        self.assertLess(jnp.max(jnp.abs(residual)), 1e-6)

    def test_solver_mode_mapping(self):
        E_spectral, phi_spectral, rho_spectral = calculate_electrostatic_fields(
            self.world,
            self.particles,
            self.constants,
            self.initial_rho,
            self.initial_phi,
            "spectral",
            "periodic",
        )

        expected_phi_spectral = solve_poisson_with_fft(rho_spectral, self.constants, self.world)
        expected_Ex_spectral, expected_Ey_spectral, expected_Ez_spectral = spectral_gradient(-1.0 * expected_phi_spectral, self.world)

        self.assertTrue(jnp.allclose(phi_spectral, expected_phi_spectral, atol=1e-8, rtol=1e-8))
        self.assertTrue(jnp.allclose(E_spectral[0], expected_Ex_spectral, atol=1e-8, rtol=1e-8))
        self.assertTrue(jnp.allclose(E_spectral[1], expected_Ey_spectral, atol=1e-8, rtol=1e-8))
        self.assertTrue(jnp.allclose(E_spectral[2], expected_Ez_spectral, atol=1e-8, rtol=1e-8))

        E_fdtd, phi_fdtd, rho_fdtd = calculate_electrostatic_fields(
            self.world,
            self.particles,
            self.constants,
            self.initial_rho,
            self.initial_phi,
            "fdtd",
            "periodic",
        )

        expected_phi_fdtd = solve_poisson_with_conjugate_gradient(
            rho_fdtd,
            self.initial_phi,
            self.constants,
            self.world,
        )
        expected_Ex_fdtd, expected_Ey_fdtd, expected_Ez_fdtd = centered_finite_difference_gradient(
            -1.0 * expected_phi_fdtd,
            self.dx,
            self.dy,
            self.dz,
            "periodic",
        )

        self.assertTrue(jnp.allclose(rho_fdtd, rho_spectral, atol=1e-12, rtol=1e-12))
        self.assertTrue(jnp.allclose(phi_fdtd, expected_phi_fdtd, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E_fdtd[0], expected_Ex_fdtd, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E_fdtd[1], expected_Ey_fdtd, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E_fdtd[2], expected_Ez_fdtd, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
