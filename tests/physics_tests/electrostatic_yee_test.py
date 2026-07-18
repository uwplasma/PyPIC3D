import unittest
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.evolve import time_loop_electrostatic
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.solvers.electrostatic_yee import (
    solve_poisson_with_conjugate_gradient,
    calculate_electrostatic_fields,
    _centered_finite_difference_gradient,
)
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from tests.kernel_fixtures import kernel_parameters, particle_species

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

        # Interior coordinate grid (no ghost cells) for analytical solutions
        x = jnp.linspace(0, self.x_wind, self.Nx, endpoint=False)
        y = jnp.linspace(0, self.y_wind, self.Ny, endpoint=False)
        z = jnp.linspace(0, self.z_wind, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')

        self.static_parameters, self.dynamic_parameters = kernel_parameters(
            Nx=self.Nx,
            Ny=self.Ny,
            Nz=self.Nz,
            x_wind=self.x_wind,
            y_wind=self.y_wind,
            z_wind=self.z_wind,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            tile_shape=(self.Nx, self.Ny, self.Nz),
            guard_cells=2,
            shape_factor=1,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            eps=1.0,
            alpha=1.0,
            electrostatic=True,
            solver="electrostatic",
        )

        self.g = int(self.static_parameters.guard_cells)
        self.active = slice(self.g, -self.g)
        # Single tile-local fields: shape (Nx+2*g, Ny+2*g, Nz+2*g).
        self.initial_rho = jnp.zeros((self.Nx + 2 * self.g, self.Ny + 2 * self.g, self.Nz + 2 * self.g))
        self.initial_phi = jnp.zeros((self.Nx + 2 * self.g, self.Ny + 2 * self.g, self.Nz + 2 * self.g))

        self.particles = [
            particle_species(
                name="test",
                charge=1.0,
                mass=1.0,
                v1=jnp.zeros(1),
                v2=jnp.zeros(1),
                v3=jnp.zeros(1),
                x1=jnp.array([0.1]),
                x2=jnp.array([0.2]),
                x3=jnp.array([0.3]),
                weight=1.0,
            )
        ]

    def test_solve_poisson_with_conjugate_gradient_single_mode(self):
        phi_true_interior = jnp.sin(self.X + self.Y + self.Z)
        rhs = apply_negative_laplacian(phi_true_interior, self.dx, self.dy, self.dz)
        rho_interior = rhs * self.dynamic_parameters.eps

        # Place rho into ghost-celled array
        rho = jnp.zeros_like(self.initial_rho)
        rho = rho.at[self.active, self.active, self.active].set(rho_interior)

        phi = solve_poisson_with_conjugate_gradient(
            rho,
            self.initial_phi,
            self.static_parameters,
            self.dynamic_parameters,
            tol=1e-10,
            max_iter=4000,
        )

        # Compare on interior
        phi_num = phi[self.active, self.active, self.active]
        phi_num = phi_num - jnp.mean(phi_num)
        phi_true = phi_true_interior - jnp.mean(phi_true_interior)

        self.assertEqual(phi_num.shape, phi_true.shape)
        self.assertTrue(jnp.allclose(phi_num, phi_true, atol=1e-7, rtol=1e-6))

        residual = apply_negative_laplacian(phi_num, self.dx, self.dy, self.dz) - rho_interior / self.dynamic_parameters.eps
        self.assertLess(jnp.max(jnp.abs(residual)), 1e-6)

    def test_electrostatic_field_solve_uses_local_centered_gradient(self):
        E, phi, rho = calculate_electrostatic_fields(
            self.static_parameters,
            self.dynamic_parameters,
            self.particles,
            self.initial_rho,
            self.initial_phi,
            "electrostatic",
            "periodic",
        )

        expected_phi = solve_poisson_with_conjugate_gradient(
            rho,
            self.initial_phi,
            self.static_parameters,
            self.dynamic_parameters,
        )
        expected_Ex, expected_Ey, expected_Ez = _centered_finite_difference_gradient(
            -1.0 * expected_phi[self.active, self.active, self.active],
            self.dx,
            self.dy,
            self.dz,
        )

        self.assertTrue(jnp.allclose(phi[self.active, self.active, self.active], expected_phi[self.active, self.active, self.active], atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E[0][self.active, self.active, self.active], expected_Ex, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E[1][self.active, self.active, self.active], expected_Ey, atol=1e-6, rtol=1e-6))
        self.assertTrue(jnp.allclose(E[2][self.active, self.active, self.active], expected_Ez, atol=1e-6, rtol=1e-6))

    def test_initialize_simulation_accepts_single_tile_electrostatic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "electrostatic init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrostatic",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "j_from_rhov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            loop, particles, fields, static_parameters, dynamic_parameters, *_ = initialize_simulation(toml.loads(toml.dumps(config)))

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(tuple(static_parameters.tile_shape), (8, 1, 1))
            for vertex_axis, center_axis in zip(dynamic_parameters.grids.vertex, dynamic_parameters.grids.center):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))
            for tiled_vertex_axis, tiled_center_axis in zip(
                dynamic_parameters.grids.tiled_vertex_grid,
                dynamic_parameters.grids.tiled_center_grid,
            ):
                self.assertTrue(jnp.allclose(tiled_vertex_axis, tiled_center_axis))
            self.assertEqual(fields[0][0].shape[:3], (1, 1, 1))
            self.assertEqual(fields[3].shape[:3], (1, 1, 1))
            self.assertEqual(fields[4].shape[:3], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
