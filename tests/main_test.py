import unittest

import jax.numpy as jnp
from main import (
    calculateE, courant_condition, plasma_frequency, debye_length,
    load_particles_from_toml, build_grid, update_parameters_from_toml,
    dump_parameters_to_toml, default_parameters, PoissonPrecondition,
    current_correction, particle_push, initialize_fields, update_B, update_E,
    autodiff_update_B, autodiff_update_E, compute_electric_divergence_error,
    compute_magnetic_divergence_error
)

class TestMain(unittest.TestCase):

    def setUp(self):
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.dt = 0.1
        self.C = 1.0
        self.eps = 8.854187817e-12
        self.kb = 1.380649e-23
        self.me = 9.10938356e-31
        self.q_e = 1.602176634e-19
        self.Te = 1.0
        self.N_electrons = 1e6
        self.x_wind = 1.0
        self.y_wind = 1.0
        self.z_wind = 1.0
        self.Nx = 10
        self.Ny = 10
        self.Nz = 10
        self.field = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        self.particles = None  # Placeholder for particles

    def test_calculateE(self):
        Ex, Ey, Ez, phi, rho = calculateE(self.particles, self.dx, self.dy, self.dz, jnp.zeros((self.Nx, self.Ny, self.Nz)), self.eps, jnp.zeros((self.Nx, self.Ny, self.Nz)), None, 0, self.x_wind, self.y_wind, self.z_wind, 'fdtd', 'periodic', False, False)
        self.assertIsNotNone(Ex)
        self.assertIsNotNone(Ey)
        self.assertIsNotNone(Ez)

    def test_courant_condition(self):
        dt = courant_condition(1, self.dx, self.dy, self.dz, self.C)
        self.assertGreater(dt, 0)

    def test_plasma_frequency(self):
        freq = plasma_frequency(self.N_electrons, self.x_wind, self.y_wind, self.z_wind, self.eps, self.me, self.q_e)
        self.assertGreater(freq, 0)

    def test_debye_length(self):
        debye = debye_length(self.eps, self.Te, self.N_electrons, self.x_wind, self.y_wind, self.z_wind, self.q_e, self.kb)
        self.assertGreater(debye, 0)

    def test_load_particles_from_toml(self):
        particles = load_particles_from_toml("config.toml", {}, self.dx, self.dy, self.dz)
        self.assertIsNotNone(particles)

    def test_build_grid(self):
        grid, staggered_grid = build_grid(self.x_wind, self.y_wind, self.z_wind, self.dx, self.dy, self.dz)
        self.assertIsNotNone(grid)
        self.assertIsNotNone(staggered_grid)

    def test_update_parameters_from_toml(self):
        simulation_parameters, plotting_parameters = update_parameters_from_toml("config.toml", {}, {})
        self.assertIsNotNone(simulation_parameters)
        self.assertIsNotNone(plotting_parameters)

    def test_dump_parameters_to_toml(self):
        dump_parameters_to_toml({}, {}, {})
        self.assertTrue(True)  # Just check if no exceptions are raised

    def test_default_parameters(self):
        plotting_parameters, simulation_parameters = default_parameters()
        self.assertIsNotNone(plotting_parameters)
        self.assertIsNotNone(simulation_parameters)

    def test_PoissonPrecondition(self):
        model = PoissonPrecondition(Nx=self.Nx, Ny=self.Ny, Nz=self.Nz, hidden_dim=3000, key=jax.random.PRNGKey(0))
        self.assertIsNotNone(model)

    def test_current_correction(self):
        Jx, Jy, Jz = current_correction(self.particles, self.Nx, self.Ny, self.Nz)
        self.assertIsNotNone(Jx)
        self.assertIsNotNone(Jy)
        self.assertIsNotNone(Jz)

    def test_particle_push(self):
        particle = particle_push(self.particles, self.field, self.field, self.field, self.field, self.field, self.field, None, None, self.dt, False)
        self.assertIsNotNone(particle)

    def test_initialize_fields(self):
        Ex, Ey, Ez, Bx, By, Bz, phi, rho = initialize_fields(self.Nx, self.Ny, self.Nz)
        self.assertIsNotNone(Ex)
        self.assertIsNotNone(Ey)
        self.assertIsNotNone(Ez)

    def test_update_B(self):
        Bx, By, Bz = update_B(None, None, self.field, self.field, self.field, self.field, self.field, self.dx, self.dy, self.dz, self.dt, 'periodic')
        self.assertIsNotNone(Bx)
        self.assertIsNotNone(By)
        self.assertIsNotNone(Bz)

    def test_update_E(self):
        Ex, Ey, Ez = update_E(None, None, self.field, self.field, self.field, self.field, self.field, self.field, self.dx, self.dy, self.dz, self.dt, self.C, self.eps, 'periodic')
        self.assertIsNotNone(Ex)
        self.assertIsNotNone(Ey)
        self.assertIsNotNone(Ez)

    def test_autodiff_update_B(self):
        Bx, By, Bz = autodiff_update_B(self.field, self.field, self.field, self.field, self.field, self.field, self.dt)
        self.assertIsNotNone(Bx)
        self.assertIsNotNone(By)
        self.assertIsNotNone(Bz)

    def test_autodiff_update_E(self):
        Ex, Ey, Ez = autodiff_update_E(self.field, self.field, self.field, self.field, self.field, self.field, self.dt, self.C)
        self.assertIsNotNone(Ex)
        self.assertIsNotNone(Ey)
        self.assertIsNotNone(Ez)

    def test_compute_electric_divergence_error(self):
        error = compute_electric_divergence_error(self.field, self.field, self.field, self.field, self.eps, self.dx, self.dy, self.dz, 'fdtd', 'periodic')
        self.assertIsNotNone(error)

    def test_compute_magnetic_divergence_error(self):
        error = compute_magnetic_divergence_error(self.field, self.field, self.field, self.dx, self.dy, self.dz, 'fdtd', 'periodic')
        self.assertIsNotNone(error)

if __name__ == '__main__':
    unittest.main()