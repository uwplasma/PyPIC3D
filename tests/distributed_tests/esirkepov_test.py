"""Focused cross-device regression tests."""

from tests.support.esirkepov_fixtures import (
    BC_ABSORBING,
    BC_PERIODIC,
    Esirkepov_current,
    TiledEsirkepovCurrentFixtures,
    TiledParticles,
    assemble_tiled_vector_field,
    compute_rho,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    os,
    refresh_tiled_particle_tiles,
    tempfile,
    toml,
    unittest,
    update_tiled_particle_positions,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):
    def test_tiled_esirkepov_honors_one_guard_cell_startup_depth(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        parameter_set["guard_cells"] = 1
        x_old, u = self._basic_positions_and_velocities(parameter_set)

        self._assert_tiled_current_matches_reference(parameter_set, x_old, u, tile_shape=(2, 1, 1))


    def test_tiled_esirkepov_matches_global_1d_periodic_current(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(parameter_set, x_old, tile_shape)

        g = int(parameter_set["guard_cells"])
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        J_tiles = Esirkepov_current(
            tiled_particles,
            update_tiled_particle_positions(tiled_particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=tiled_particles.u,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=g)
        _, J_reference = self._assembled_esirkepov_current(
            parameter_set,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            dynamic_values,
            self._one_tile_shape_for_parameters(parameter_set),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )


    def test_tiled_esirkepov_matches_global_current_for_dimensions_and_shapes(self):
        for shape_factor in (1, 2):
            cases = ((4, 4, 4),)
            for Nx, Ny, Nz in cases:
                with self.subTest(shape_factor=shape_factor, shape=(Nx, Ny, Nz)):
                    parameter_set = self._build_parameter_values(Nx=Nx, Ny=Ny, Nz=Nz, dt=0.05, shape_factor=shape_factor)
                    active_axes = sum(int(width > 1) for width in (Nx, Ny, Nz))
                    # The production tiled Yee startup promotes Esirkepov current storage to two guards.
                    if shape_factor == 2 or active_axes == 3:
                        parameter_set["guard_cells"] = 2
                    x_old, u = self._basic_positions_and_velocities(parameter_set)
                    self._assert_tiled_current_matches_reference(parameter_set, x_old, u)


    def test_tiled_esirkepov_folds_internal_tile_and_periodic_boundary_crossings(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05)
        dx = parameter_set["dx"]
        dt = parameter_set["dt"]
        x_old = jnp.array(
            [
                [-1.0 * dx, 0.0, 0.0],
                [0.5 * parameter_set["x_wind"] - 0.2 * dx, 0.0, 0.0],
                [-0.5 * parameter_set["x_wind"] + 0.2 * dx, 0.0, 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.7 * dx / dt, 0.0, 0.0],
                [0.6 * dx / dt, 0.0, 0.0],
                [-0.6 * dx / dt, 0.0, 0.0],
            ]
        )

        self._assert_tiled_current_matches_reference(parameter_set, x_old, u, tile_shape=(2, 1, 1))


    def test_tiled_esirkepov_uses_particle_boundaries_for_current_ghosts(self):
        tile_shape = (2, 1, 1)
        periodic_particle_parameters = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={"x": 0, "y": 0, "z": 0},
        )
        absorbing_particle_parameters = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={
                "x": BC_ABSORBING,
                "y": BC_PERIODIC,
                "z": BC_PERIODIC,
            },
        )
        periodic_particle_parameters["guard_cells"] = 2
        absorbing_particle_parameters["guard_cells"] = 2
        periodic_particle_parameters = self._parameters_with_tiled_grids(periodic_particle_parameters, tile_shape)
        absorbing_particle_parameters = self._parameters_with_tiled_grids(absorbing_particle_parameters, tile_shape)

        dx = periodic_particle_parameters["dx"]
        dt = periodic_particle_parameters["dt"]
        x_old = jnp.array([[1.75, 0.0, 0.0], [-1.75, 0.0, 0.0]])
        u = jnp.array([[0.6 * dx / dt, 0.0, 0.0], [-0.6 * dx / dt, 0.0, 0.0]])
        particles, species_config = self._particles_from_arrays(periodic_particle_parameters, tile_shape, x_old, u)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}

        _, _, J_template, _, _ = self._initialize_fields(periodic_particle_parameters, dynamic_values)
        static_periodic, dynamic_periodic = kernel_parameters_from_values(periodic_particle_parameters, dynamic_values)
        static_absorbing, dynamic_absorbing = kernel_parameters_from_values(absorbing_particle_parameters, dynamic_values)

        periodic_bc_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_periodic.dt),
            species_config,
            J_template,
            static_periodic,
            dynamic_periodic,
            coordinate_velocity=particles.u,
        )
        absorbing_bc_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_absorbing.dt),
            species_config,
            J_template,
            static_absorbing,
            dynamic_absorbing,
            coordinate_velocity=particles.u,
        )

        max_difference = max(
            float(jnp.max(jnp.abs(periodic_component - absorbing_component)))
            for periodic_component, absorbing_component in zip(periodic_bc_current, absorbing_bc_current)
        )
        self.assertGreater(max_difference, 1.0e-12)


    def test_tiled_esirkepov_satisfies_tile_local_discrete_continuity(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dx = parameter_set["dx"]
        x_old = jnp.array(
            [
                [-1.0 * dx, 0.0, 0.0],
                [0.5 * parameter_set["x_wind"] - 0.2 * dx, 0.0, 0.0],
                [0.25 * parameter_set["x_wind"], 0.0, 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.7 * dx / parameter_set["dt"], 0.0, 0.0],
                [0.6 * dx / parameter_set["dt"], 0.0, 0.0],
                [-0.4 * dx / parameter_set["dt"], 0.0, 0.0],
            ]
        )
        tiled_particles, species_config = self._particles_from_arrays(parameter_set, tile_shape, x_old, u)
        g = int(parameter_set["guard_cells"])
        rho_tiles = self._build_tiled_array(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)

        rho_old = compute_rho(tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        J_tiles = Esirkepov_current(
            tiled_particles,
            update_tiled_particle_positions(tiled_particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=tiled_particles.u,
        )
        new_particles = update_tiled_particle_positions(tiled_particles, species_config, parameter_set["dt"])
        new_particles, overflow = refresh_tiled_particle_tiles(new_particles, static_parameters, dynamic_parameters)
        rho_new = compute_rho(new_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        drhodt = (rho_new[:, :, :, g:-g, g:-g, g:-g] - rho_old[:, :, :, g:-g, g:-g, g:-g]) / parameter_set["dt"]
        dJxdx = (J_tiles[0][:, :, :, g:-g, g:-g, g:-g] - J_tiles[0][:, :, :, g - 1:-g - 1, g:-g, g:-g]) / parameter_set["dx"]
        continuity = drhodt + dJxdx
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(drhodt)) + jnp.max(jnp.abs(dJxdx)))

        self.assertLessEqual(float(jnp.max(jnp.abs(continuity))), float(1.0e-12 * scale))


    def test_initialize_tiled_yee_esirkepov_uses_two_guard_current_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            jnp.save(x_path, jnp.array([-1.5, -0.5, 0.5, 1.5]))
            jnp.save(zeros_path, jnp.zeros(4))
            jnp.save(vx_path, jnp.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "guard_cells": 2,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
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
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            _loop, particles, fields, static_parameters, *_rest = initialize_simulation(toml.load(config_path))
            E_tiles, _B_tiles, J_tiles, *_ = fields

            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(E_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(J_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(int(static_parameters.guard_cells), 2)
            self.assertFalse(hasattr(static_parameters, "current_guard_cells"))
            self.assertEqual(static_parameters.current_deposition, "esirkepov")
            self.assertEqual(static_parameters.current_filter, "none")


    def test_tiled_yee_esirkepov_loop_advances_particles_once_before_retiling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_initial = jnp.array([-1.5, -0.5, 0.5, 1.5])
            vx_initial = jnp.array([0.10, -0.05, 0.07, -0.02])
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            jnp.save(x_path, x_initial)
            jnp.save(zeros_path, jnp.zeros(4))
            jnp.save(vx_path, vx_initial)

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov step smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
                    "current_calculation": "esirkepov",
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
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov_step.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            (
                loop,
                particles,
                fields,
                static_parameters,
                dynamic_parameters,
                _plotting_parameters,
                _plasma_parameters,
                species_config,
            ) = initialize_simulation(toml.load(config_path))

            particles, fields = loop(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

            active_x = jnp.asarray(particles.x[..., 0][particles.active])
            expected_x = jnp.sort(x_initial + vx_initial * float(dynamic_parameters.dt))
            self.assertTrue(jnp.allclose(jnp.sort(active_x), expected_x, rtol=1.0e-12, atol=1.0e-12))
            self.assertEqual(int(static_parameters.guard_cells), 2)
            self.assertEqual(fields[2][0].shape[-3:], (6, 5, 5))
            self.assertFalse(bool(fields[-1]))


    def test_tiled_yee_esirkepov_staging_uses_parameter_contract_not_function_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_initial = jnp.array([-1.5, -0.5, 0.5, 1.5])
            vx_initial = jnp.array([0.10, -0.05, 0.07, -0.02])
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            jnp.save(x_path, x_initial)
            jnp.save(zeros_path, jnp.zeros(4))
            jnp.save(vx_path, vx_initial)

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov alias staging",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
                    "current_calculation": "esirkepov",
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

            (
                loop,
                particles,
                fields,
                static_parameters,
                dynamic_parameters,
                _plotting_parameters,
                _plasma_parameters,
                species_config,
            ) = initialize_simulation(config)
            self.assertEqual(static_parameters.current_deposition, "esirkepov")
            initial_particles = particles
            initial_fields = fields
            particles, fields = loop(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

            reference_J = Esirkepov_current(
                initial_particles,
                update_tiled_particle_positions(initial_particles, species_config, dynamic_parameters.dt),
                species_config,
                initial_fields[2],
                static_parameters,
                dynamic_parameters,
                coordinate_velocity=initial_particles.u,
            )

            for reference_component, tiled_component in zip(reference_J, fields[2]):
                self.assertTrue(
                    jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                    f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
                )


if __name__ == "__main__":
    unittest.main()
