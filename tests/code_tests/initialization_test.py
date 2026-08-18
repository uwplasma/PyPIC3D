import unittest
import tempfile
import os
from unittest.mock import patch

import numpy as np
import toml
import jax
import jax.numpy as jnp
from PyPIC3D.initialization import (
    _encode_field_bc,
    _encode_particle_bc,
    default_parameters,
    initialize_simulation,
    setup_write_dir,
    validate_field_solver,
)
from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
)
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import build_yee_grid

jax.config.update("jax_enable_x64", True)

class TestInitializationFunctions(unittest.TestCase):
    def setUp(self):
        self.plotting_parameters, self.simulation_parameters, self.dynamic_values = default_parameters()
        self.simulation_parameters['output_dir'] = 'test_output'
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
        deprecated_plotting_flags = (
            "plotting",
            "save_data",
            "plotfields",
            "plotpositions",
            "plotenergy",
            "plotcurrent",
            "plasmaFreq",
            "plot_phasespace",
            "plot_errors",
            "plot_dispersion",
            "plot_chargeconservation",
        )
        for flag in deprecated_plotting_flags:
            self.assertNotIn(flag, plotting)
        self.assertFalse(plotting["plotchargedensity"])
        self.assertIn('eps', dynamic)
        # check that the default parameters contain expected keys

    def test_encode_field_bc_accepts_constant_boundary(self):
        self.assertEqual(_encode_field_bc("constant"), BC_CONSTANT)

    def test_field_and_particle_boundaries_use_one_code_map(self):
        self.assertEqual(BC_PERIODIC, 0)
        self.assertEqual(BC_CONDUCTING, 1)
        self.assertEqual(BC_ABSORBING, 2)
        self.assertEqual(BC_CONSTANT, 3)

        self.assertEqual(_encode_field_bc("periodic"), BC_PERIODIC)
        self.assertEqual(_encode_field_bc("conducting"), BC_CONDUCTING)
        self.assertEqual(_encode_field_bc("constant"), BC_CONSTANT)
        self.assertEqual(_encode_particle_bc("periodic"), BC_PERIODIC)
        self.assertEqual(_encode_particle_bc("reflecting"), BC_CONDUCTING)
        self.assertEqual(_encode_particle_bc("absorbing"), BC_ABSORBING)

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
                "plotting": {
                    "dump_fields": True,
                    "plotchargedensity": True,
                },
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

            with patch("PyPIC3D.initialization.write_openpmd_initial_fields") as write_initial_fields:
                (
                    loop,
                    particles,
                    fields,
                    parameter_set,
                    dynamic_parameters,
                    plotting_parameters,
                    *_rest,
                ) = initialize_simulation(toml.load(config_path))

            self.assertIs(loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(particles.x.sharding.mesh, parameter_set.field_mesh)
            self.assertEqual(particles.active.sharding.mesh, parameter_set.field_mesh)
            self.assertEqual(len(particles.x.addressable_shards), 2)
            self.assertEqual(parameter_set.solver, "electrodynamic_yee")
            self.assertEqual(tuple(parameter_set.tile_shape), (2, 1, 1))
            self.assertNotIn("particle_species_names", parameter_set)
            self.assertNotIn("particle_species_metadata", parameter_set)
            self.assertEqual(plotting_parameters["particle_species_names"], ("electrons",))
            self.assertEqual(plotting_parameters["particle_species_metadata"][0]["name"], "electrons")
            self.assertEqual(tuple(plotting_parameters["field_map"]), ("E", "B", "J", "rho"))
            self.assertEqual(tuple(write_initial_fields.call_args.args[0]), ("E", "B", "J", "rho"))
            self.assertTrue(jnp.any(plotting_parameters["field_map"]["rho"] != 0.0))
            self.assertIn("tiled_center_grid", dynamic_parameters.grids._asdict())
            self.assertIn("tiled_vertex_grid", dynamic_parameters.grids._asdict())
            expected_center_grid, expected_vertex_grid = build_yee_grid(dynamic_parameters)
            for axis, expected_axis in zip(dynamic_parameters.grids.center, expected_center_grid):
                self.assertTrue(jnp.allclose(axis, expected_axis))
            for axis, expected_axis in zip(dynamic_parameters.grids.vertex, expected_vertex_grid):
                self.assertTrue(jnp.allclose(axis, expected_axis))

            g = int(parameter_set.guard_cells)
            for axis_index, (tiled_axis, expected_axis, tile_width) in enumerate(
                zip(dynamic_parameters.grids.tiled_center_grid, expected_center_grid, parameter_set.tile_shape)
            ):
                for tile_index in range(int(dynamic_parameters.grids.tiled_center_grid[axis_index].shape[axis_index])):
                    tile_slice = [0, 0, 0, slice(g, -g)]
                    tile_slice[axis_index] = tile_index
                    start = 1 + tile_index * int(tile_width)
                    stop = start + int(tile_width)
                    self.assertTrue(jnp.allclose(tiled_axis[tuple(tile_slice)], expected_axis[start:stop]))
            for axis_index, (tiled_axis, expected_axis, tile_width) in enumerate(
                zip(dynamic_parameters.grids.tiled_vertex_grid, expected_vertex_grid, parameter_set.tile_shape)
            ):
                for tile_index in range(int(dynamic_parameters.grids.tiled_vertex_grid[axis_index].shape[axis_index])):
                    tile_slice = [0, 0, 0, slice(g, -g)]
                    tile_slice[axis_index] = tile_index
                    start = 1 + tile_index * int(tile_width)
                    stop = start + int(tile_width)
                    self.assertTrue(jnp.allclose(tiled_axis[tuple(tile_slice)], expected_axis[start:stop]))
            E, B, J, rho, phi, external_fields, pml_state, overflow = fields
            self.assertEqual(E[0].ndim, 6)
            self.assertEqual(B[0].ndim, 6)
            self.assertEqual(J[0].ndim, 6)
            self.assertIsNone(pml_state)
            self.assertFalse(bool(overflow))
            # dump a dummy config file to tmp directory and confirm it can be read
            # in correctly

    def test_initialize_static_metric_loads_previous_fields_from_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_dtheta_path = os.path.join(tmpdir, "current_dtheta.npy")
            previous_dtheta_path = os.path.join(tmpdir, "previous_dtheta.npy")
            current_bphi_path = os.path.join(tmpdir, "current_bphi.npy")
            previous_bphi_path = os.path.join(tmpdir, "previous_bphi.npy")
            np.save(current_dtheta_path, np.full((4, 4, 1), 2.0))
            np.save(previous_dtheta_path, np.full((4, 4, 1), 3.0))
            np.save(current_bphi_path, np.full((4, 4, 1), 5.0))
            np.save(previous_bphi_path, np.full((4, 4, 1), 7.0))
            config = {
                "simulation_parameters": {
                    "name": "static previous field load test",
                    "output_dir": tmpdir,
                    "solver": "static_metric",
                    "metric": "flat_spherical",
                    "particle_pusher": "hybrid_boris_geodesic",
                    "current_calculation": "GR_direct_deposition",
                    "Nx": 4,
                    "Ny": 4,
                    "Nz": 1,
                    "x_min": 1.0,
                    "x_max": 2.0,
                    "y_min": 0.1,
                    "y_max": 2 * np.pi + 0.1,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-2,
                    "particle_tile_nx": 4,
                    "particle_tile_ny": 4,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                    "x_bc": "constant",
                    "y_bc": "periodic",
                    "z_bc": "periodic",
                    "C": 1.0,
                    "eps": 1.0,
                    "mu": 1.0,
                },
                "plotting": {"plotting": False},
                "field1": {"name": "Dtheta", "type": 1, "path": current_dtheta_path},
                "field2": {"name": "Bphi", "type": 5, "path": current_bphi_path},
                "previous_field1": {"name": "Dtheta previous", "type": 1, "path": previous_dtheta_path},
                "previous_field2": {"name": "Bphi previous", "type": 5, "path": previous_bphi_path},
            }

            loop, particles, fields, static_parameters, dynamic_parameters, *_rest = initialize_simulation(config)

            self.assertIs(loop, time_loop_static_metric)
            g = int(static_parameters.guard_cells)
            D, B, _J, _rho, _phi, _external_fields, _metric, previous_fields, _overflow = fields
            D_previous, B_previous = previous_fields
            interior = (0, 0, 0, slice(g, -g), slice(g, -g), slice(g, -g))
            self.assertTrue(jnp.allclose(D[1][interior], 2.0))
            self.assertTrue(jnp.allclose(B[2][interior], 5.0))
            self.assertTrue(jnp.allclose(D_previous[1][interior], 3.0))
            self.assertTrue(jnp.allclose(B_previous[2][interior], 7.0))

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
                "plotting": {
                    "plotvelocities": True,
                },
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
            plotting_parameters = result[5]

            self.assertGreater(float(dynamic_parameters.dt), 0.0)
            self.assertEqual(
                tuple(plotting_parameters["field_map"]),
                ("E", "B", "J", "fluid_velocity"),
            )
            for velocity_component in plotting_parameters["field_map"]["fluid_velocity"]:
                self.assertTrue(jnp.allclose(velocity_component, 0.0))

    def test_initialize_simulation_builds_grid_from_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            z_path = os.path.join(tmpdir, "z.npy")
            np.save(x_path, np.array([1.25, 1.75, 2.25, 2.75]))
            np.save(zeros_path, np.zeros(4))
            np.save(z_path, np.full(4, 2.5))
            config = {
                "simulation_parameters": {
                    "name": "shifted grid bounds test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_min": 1.0,
                    "x_max": 3.0,
                    "y_min": -0.5,
                    "y_max": 0.5,
                    "z_min": 2.0,
                    "z_max": 3.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
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
                    "initial_z": z_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            result = initialize_simulation(config)
            dynamic_parameters = result[4]

            self.assertAlmostEqual(float(dynamic_parameters.dx), 0.5)
            self.assertAlmostEqual(float(dynamic_parameters.x_wind), 2.0)
            self.assertTrue(jnp.allclose(
                dynamic_parameters.grids.center[0],
                jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            ))
            self.assertTrue(jnp.allclose(
                dynamic_parameters.grids.vertex[0],
                jnp.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25]),
            ))
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[1][1]), -0.5)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[1][-1]), 0.5)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[2][1]), 2.0)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[2][-1]), 3.0)

    def test_initialize_simulation_rejects_one_sided_grid_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "simulation_parameters": {
                    "name": "one sided bounds test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_min": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
                "plotting": {"plotting": False},
            }

            with self.assertRaisesRegex(ValueError, "Both x_min and x_max"):
                initialize_simulation(config)

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

            self.assertEqual(
                parameter_set.particle_boundary_conditions,
                (BC_CONDUCTING, BC_ABSORBING, BC_PERIODIC),
            )
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
                "plotting": {
                    "dump_fields": True,
                },
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

            with patch("PyPIC3D.initialization.write_openpmd_initial_fields") as write_initial_fields:
                (
                    loop,
                    particles,
                    fields,
                    parameter_set,
                    dynamic_parameters,
                    plotting_parameters,
                    *_rest,
                ) = initialize_simulation(config)

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(tuple(plotting_parameters["field_map"]), ("rho", "phi", "E"))
            self.assertEqual(tuple(write_initial_fields.call_args.args[0]), ("rho", "phi", "E"))
            self.assertTrue(jnp.any(plotting_parameters["field_map"]["rho"] != 0.0))
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
