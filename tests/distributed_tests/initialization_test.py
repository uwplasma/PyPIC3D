"""Focused cross-device regression tests."""

from tests.support.initialization_fixtures import (
    InitializationFunctionsFixtures,
    TiledParticles,
    build_yee_grid,
    initialize_simulation,
    jnp,
    np,
    os,
    patch,
    tempfile,
    time_loop_electrodynamic,
    toml,
    unittest,
)


class TestInitializationFunctions(InitializationFunctionsFixtures, unittest.TestCase):
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
            self.assertEqual(parameter_set.particle_batch_size, 2)
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


if __name__ == "__main__":
    unittest.main()
