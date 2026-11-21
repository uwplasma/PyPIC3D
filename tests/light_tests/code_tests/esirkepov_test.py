"""One-tile API contracts."""

from tests.support.esirkepov_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    Esirkepov_current,
    SpeciesConfig,
    TiledEsirkepovCurrentFixtures,
    TiledParticles,
    assemble_tiled_vector_field,
    build_static_parameters,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    os,
    tempfile,
    tile_vector_field,
    unittest,
    update_E,
    update_tiled_particle_positions,
    vector_field_for_output,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):
    def test_initialize_fields_builds_tiled_current_with_startup_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        parameter_set["guard_cells"] = 2
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["tile_shape"] = tile_shape

        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set)

        self.assertEqual(J_tiles[0].shape, (1, 1, 1, 12, 5, 5))
        self.assertTrue(jnp.allclose(J_tiles[0], 0.0))

    def test_shared_guard_current_tiles_assemble_to_one_guard_global_current(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        parameter_set["guard_cells"] = 2
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["tile_shape"] = tile_shape
        g = int(parameter_set["guard_cells"])
        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[0, 0, 0, 4, 2, 2].set(3.0)

        assembled = assemble_tiled_vector_field((Jx, Jy, Jz), parameter_set, tile_shape, num_guard_cells=g)

        self.assertEqual(assembled[0].shape, (10, 3, 3))
        self.assertEqual(float(assembled[0][3, 1, 1]), 3.0)


    def test_tiled_esirkepov_zeroes_disabled_current_in_reduced_dimensions(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05)
        parameter_set["guard_cells"] = 2
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        particles = TiledParticles(
            x=jnp.asarray([[[[[[0.0, 0.0, 0.0]]]]]]),
            u=jnp.asarray([[[[[[0.3, 0.4, 0.5]]]]]]),
            active=jnp.asarray([[[[[True]]]]]),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.zeros((1, 3), dtype=bool),
        )
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(
            parameter_set,
            dynamic_values,
        )

        current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=particles.u,
        )

        for component in current:
            self.assertTrue(jnp.allclose(component, 0.0))


    def test_esirkepov_current_refresh_uses_reflecting_vector_parity(self):
        parameter_set = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=4,
            dt=0.05,
            shape_factor=2,
            boundary_conditions={
                "x": BC_PERIODIC,
                "y": BC_PERIODIC,
                "z": BC_CONDUCTING,
            },
            particle_boundary_conditions={
                "x": BC_PERIODIC,
                "y": BC_PERIODIC,
                "z": BC_CONDUCTING,
            },
        )
        parameter_set["guard_cells"] = 2
        tile_shape = (4, 1, 4)
        x_old = jnp.array(
            [
                [-0.2, 0.0, -1.99],
                [0.2, 0.0, 1.99],
            ]
        )
        u = jnp.array(
            [
                [0.3, -0.2, 0.25],
                [-0.1, 0.4, -0.15],
            ]
        )
        particles, species_config = self._particles_from_arrays(
            parameter_set,
            tile_shape,
            x_old,
            u,
        )
        current_tiles, _ = self._assembled_esirkepov_current(
            parameter_set,
            particles,
            species_config,
            {"C": 1.0, "eps": 1.0, "alpha": 1.0},
            tile_shape,
        )
        g = int(parameter_set["guard_cells"])
        n = tile_shape[2]

        # Jx and Jy are tangential to the z-walls and collocated in z: even
        # images about the wall nodes g and g+n, so ghost g-k mirrors g+k.
        for component in (0, 1):
            current = current_tiles[component][0, 0, 0]
            self.assertGreater(float(jnp.max(jnp.abs(current[g:-g, g:-g, g:-g]))), 0.0)
            for k in range(1, g + 1):
                self.assertTrue(jnp.allclose(current[g:-g, g:-g, g - k], current[g:-g, g:-g, g + k], atol=1.0e-12))
            for k in range(1, g):
                self.assertTrue(jnp.allclose(current[g:-g, g:-g, g + n + k], current[g:-g, g:-g, g + n - k], atol=1.0e-12))
        # Jz is normal and staggered in z: odd images about the wall faces
        current = current_tiles[2][0, 0, 0]
        self.assertGreater(float(jnp.max(jnp.abs(current[g:-g, g:-g, g:-g]))), 0.0)
        self.assertTrue(jnp.allclose(current[g:-g, g:-g, :g],
                                     -jnp.flip(current[g:-g, g:-g, g:2 * g], axis=-1), atol=1.0e-12))
        self.assertTrue(jnp.allclose(current[g:-g, g:-g, -g:],
                                     -jnp.flip(current[g:-g, g:-g, -2 * g:-g], axis=-1), atol=1.0e-12))

    def test_initialize_rejects_filtered_esirkepov_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            jnp.save(x_path, jnp.array([-1.5, -0.5, 0.5, 1.5]))
            jnp.save(zeros_path, jnp.zeros(4))
            jnp.save(vx_path, jnp.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "reject filtered esirkepov",
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
                    "particle_tile_nx": 8,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "digital",
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

            with self.assertRaisesRegex(ValueError, "Esirkepov current filtering is not supported"):
                initialize_simulation(config)


if __name__ == "__main__":
    unittest.main()
