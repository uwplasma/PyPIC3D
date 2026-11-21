"""One-tile API contracts."""


from tests.support.initialization_fixtures import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    InitializationFunctionsFixtures,
    TiledParticles,
    _available_cpu_threads,
    _encode_field_bc,
    _encode_particle_bc,
    _resolve_particle_batch_size,
    build_static_parameters,
    default_parameters,
    initialize_simulation,
    jnp,
    kernel_parameters,
    np,
    os,
    patch,
    physical_fields,
    setup_write_dir,
    tempfile,
    time_loop_electrostatic,
    time_loop_static_metric,
    toml,
    unittest,
    update_parameters_from_toml,
    validate_field_solver,
)


class TestInitializationFunctions(InitializationFunctionsFixtures, unittest.TestCase):


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


if __name__ == "__main__":
    unittest.main()
