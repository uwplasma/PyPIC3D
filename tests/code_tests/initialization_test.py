import unittest
import tempfile
import os
import numpy as np
import toml
import jax
import jax.numpy as jnp
from PyPIC3D.initialization import setup_write_dir, default_parameters, initialize_simulation, validate_field_solver
from PyPIC3D.evolve import time_loop_electrodynamic, time_loop_electrostatic
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import build_yee_grid

jax.config.update("jax_enable_x64", True)

class TestInitializationFunctions(unittest.TestCase):
    def setUp(self):
        self.plotting_parameters, self.simulation_parameters, self.dynamic_values = default_parameters()
        self.simulation_parameters['output_dir'] = 'test_output'
        self.plotting_parameters['plotfields'] = False
        # check the  default parameters are set correctly

    def test_setup_write_dir(self):
        # Should not raise
        setup_write_dir(self.simulation_parameters, self.plotting_parameters)
        # check that the output directory is created

    def test_default_parameters(self):
        plotting, sim, dynamic = default_parameters()
        self.assertIn('Nx', dynamic)
        self.assertIn('particle_pusher', sim)
        self.assertEqual(sim['particle_pusher'], 'boris')
        self.assertEqual(sim["solver"], "electrodynamic_yee")
        self.assertNotIn("electrostatic", sim)
        self.assertNotIn("fast_backend", sim)
        self.assertEqual(sim["particle_x_bc"], "periodic")
        self.assertEqual(sim["particle_y_bc"], "periodic")
        self.assertEqual(sim["particle_z_bc"], "periodic")
        self.assertEqual(sim["guard_cells"], 2)
        self.assertNotIn("plot_vtk_particles", plotting)
        self.assertNotIn("plot_vtk_scalars", plotting)
        self.assertNotIn("plot_vtk_vectors", plotting)
        self.assertIn('eps', dynamic)
        self.assertIn('plotfields', plotting)
        # check that the default parameters contain expected keys

    def test_initialize_simulation_returns_tiled_runtime_for_ordinary_electrodynamic_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            np.save(x_path, np.array([-0.375, -0.125, 0.125, 0.375]))
            np.save(zeros_path, np.zeros(4))
            config = {
                "simulation_parameters": {
                    "name": "ordinary tiled runtime test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                },
                "plotting": {"plotting": False},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            config_path = os.path.join(tmpdir, "global_particle_bc.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, parameter_set, dynamic_parameters, plotting_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(parameter_set.solver, "electrodynamic_yee")
            self.assertEqual(tuple(parameter_set.tile_shape), (2, 1, 1))
            self.assertNotIn("particle_species_names", parameter_set)
            self.assertNotIn("particle_species_metadata", parameter_set)
            self.assertEqual(plotting_parameters["particle_species_names"], ("electrons",))
            self.assertEqual(plotting_parameters["particle_species_metadata"][0]["name"], "electrons")
            self.assertIn("tiled_center_grid", dynamic_parameters.grids._asdict())
            self.assertIn("tiled_vertex_grid", dynamic_parameters.grids._asdict())
            expected_vertex_grid, expected_center_grid = build_yee_grid(dynamic_parameters)
            for axis, expected_axis in zip(dynamic_parameters.grids.vertex, expected_vertex_grid):
                self.assertTrue(jnp.allclose(axis, expected_axis))
            for axis, expected_axis in zip(dynamic_parameters.grids.center, expected_center_grid):
                self.assertTrue(jnp.allclose(axis, expected_axis))
            E, B, J, rho, phi, external_fields, pml_state, overflow = fields
            self.assertEqual(E[0].ndim, 6)
            self.assertEqual(B[0].ndim, 6)
            self.assertEqual(J[0].ndim, 6)
            self.assertIsNone(pml_state)
            self.assertFalse(bool(overflow))
            # dump a dummy config file to tmp directory and confirm it can be read
            # in correctly

    def test_initialize_simulation_computes_courant_dt_before_runtime_parameters_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            np.save(x_path, np.array([-0.375, -0.125, 0.125, 0.375]))
            np.save(zeros_path, np.zeros(4))
            config = {
                "simulation_parameters": {
                    "name": "courant dt tiled runtime test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "particle_tile_nx": 4,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                },
                "plotting": {"plotting": False},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            config_path = os.path.join(tmpdir, "courant_dt.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            result = initialize_simulation(toml.load(config_path))
            dynamic_parameters = result[4]

            self.assertGreater(float(dynamic_parameters.dt), 0.0)

    def test_initialize_simulation_encodes_global_particle_boundary_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            np.save(zeros_path, np.zeros(1))
            config = {
                "simulation_parameters": {
                    "name": "global particle bc test",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 1,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                    "particle_x_bc": "reflecting",
                    "particle_y_bc": "absorbing",
                    "particle_z_bc": "periodic",
                },
                "plotting": {"plotting": False},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 1,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "x_bc": "absorbing",
                    "initial_x": zeros_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            config_path = os.path.join(tmpdir, "global_particle_bc.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            _, particles, _, parameter_set, *_ = initialize_simulation(toml.load(config_path))

            self.assertEqual(parameter_set.particle_boundary_conditions, (1, 2, 0))
            self.assertIsInstance(particles, TiledParticles)
            # check that the global particle boundary conditions are encoded correctly in the parameter_set dictionary

    def test_initialize_simulation_uses_collocated_grid_for_electrostatic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            np.save(zeros_path, np.zeros(1))
            config = {
                "simulation_parameters": {
                    "name": "electrostatic collocated grid test",
                    "output_dir": tmpdir,
                    "solver": "electrostatic",
                    "Nx": 4,
                    "Ny": 2,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
                "plotting": {"plotting": False},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 1,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": zeros_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            loop, particles, fields, parameter_set, dynamic_parameters, *_ = initialize_simulation(config)

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            for vertex_axis, center_axis in zip(dynamic_parameters.grids.vertex, dynamic_parameters.grids.center):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))
        # test the initialize_simulation function with an electrostatic solver and check that it uses a collocated grid

    def test_initialize_simulation_rejects_unknown_solver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "simulation_parameters": {
                    "name": "unknown solver test",
                    "output_dir": tmpdir,
                    "solver": "old_solver",
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
                "plotting": {"plotting": False},
            }

            with self.assertRaisesRegex(ValueError, "Unsupported solver"):
                initialize_simulation(config)
        # test that initialize_simulation raises an error for an unknown solver

    def test_validate_field_solver_rejects_spectral(self):
        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("spectral")

    def test_validate_field_solver_accepts_only_public_runtime_modes(self):
        validate_field_solver("electrodynamic_yee")
        validate_field_solver("electrostatic")

        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("fdtd")
        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("tiled_yee")

if __name__ == '__main__':
    unittest.main()
