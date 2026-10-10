"""Single-device numerical tests."""

from tests.support.esirkepov_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    Esirkepov_current,
    SpeciesConfig,
    TiledEsirkepovCurrentFixtures,
    TiledParticles,
    assemble_tiled_vector_field,
    build_static_parameters,
    compute_rho,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    os,
    refresh_tiled_particle_tiles,
    tempfile,
    tile_vector_field,
    unittest,
    update_E,
    update_tiled_particle_positions,
    vector_field_for_output,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):
    def test_esirkepov_continuity_for_dimensions_and_shapes(self):
        for shape_factor in (1, 2):
            two_dimensional_case = (8, 8, 1) if shape_factor == 1 else (1, 8, 8)
            cases = (
                (8, 1, 1),
                (1, 8, 1),
                (1, 1, 8),
                two_dimensional_case,
                (4, 4, 4),
            )
            for Nx, Ny, Nz in cases:
                with self.subTest(shape_factor=shape_factor, shape=(Nx, Ny, Nz)):
                    parameter_set = self._build_parameter_values(Nx=Nx, Ny=Ny, Nz=Nz, dt=0.05, shape_factor=shape_factor)
                    active_axes = sum(int(width > 1) for width in (Nx, Ny, Nz))
                    # The production tiled Yee startup promotes Esirkepov current storage to two guards.
                    if shape_factor == 2 or active_axes == 3:
                        parameter_set["guard_cells"] = 2
                    x_old, u = self._basic_positions_and_velocities(parameter_set)
                    self._assert_single_tile_continuity(parameter_set, x_old, u)


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


    def test_output_adapter_uses_parameter_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        parameter_set["guard_cells"] = 2
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["tile_shape"] = tile_shape
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        g = int(parameter_set["guard_cells"])
        E_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[0, 0, 0, 4, 2, 2].set(7.0)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        fields = (E_tiles, B_tiles, (Jx, Jy, Jz), rho, phi, (E_tiles, B_tiles), None)

        static_parameters = build_static_parameters(parameter_set)
        output_current = vector_field_for_output(fields[2], static_parameters)

        self.assertEqual(output_current[0].shape, shape)
        self.assertEqual(float(output_current[0][3, 1, 1]), 7.0)
        self.assertEqual(fields[2][0].shape[-3:], (12, 5, 5))


    def test_update_E_reads_two_guard_current_interior(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=1, Nz=1, dt=0.25)
        dynamic_values = {"C": 1.0, "eps": 2.0}
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set["guard_cells"] = 2
        g = int(parameter_set["guard_cells"])
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        E_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, parameter_set, tile_shape, num_guard_cells=g)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        _, _, J_tiles, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 2:-2, 2:-2, 2:-2].set(4.0)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E_after, pml_state = update_E(E_tiles, B_tiles, (Jx, Jy, Jz), static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertTrue(jnp.allclose(E_after[0][:, :, :, g:-g, g:-g, g:-g], -0.5))


    def test_tiled_esirkepov_masks_current_per_species_direction(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=4, Nz=4, dt=0.05)
        parameter_set["guard_cells"] = 2
        tile_shape = (4, 4, 4)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        x = jnp.asarray(
            [[[[[[-0.75, -0.25, 0.25]], [[0.75, 0.25, -0.25]]]]]]
        )
        u = jnp.asarray(
            [[[[[[0.2, 0.3, 0.4]], [[-0.5, -0.6, -0.7]]]]]]
        )
        particles = TiledParticles(
            x=x,
            u=u,
            active=jnp.ones(x.shape[:-1], dtype=bool),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([1.0, 2.0]),
            mass=jnp.asarray([1.0, 1.0]),
            weight=jnp.asarray([1.0, 1.0]),
            update_x=jnp.asarray([
                (False, True, False),
                (True, False, True),
            ]),
        )
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(
            parameter_set,
            dynamic_values,
        )

        masked_current = Esirkepov_current(
            particles,
            update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=particles.u,
        )

        slot_mask = species_config.update_x.reshape((1, 1, 1, 2, 1, 3))
        reference_particles = particles._replace(u=jnp.where(slot_mask, particles.u, 0.0))
        reference_config = species_config._replace(update_x=jnp.ones_like(species_config.update_x))
        reference_current = Esirkepov_current(
            reference_particles,
            update_tiled_particle_positions(reference_particles, reference_config, dynamic_parameters.dt),
            reference_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=reference_particles.u,
        )

        for masked_component, reference_component in zip(masked_current, reference_current):
            self.assertTrue(jnp.allclose(masked_component, reference_component))
            self.assertGreater(float(jnp.max(jnp.abs(masked_component))), 0.0)


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


    def test_public_Esirkepov_current_dispatches_tiled_particles_to_tile_local_current(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        parameter_set["guard_cells"] = 2
        parameter_set = self._parameters_with_tiled_grids(parameter_set, (8, 1, 1))
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(parameter_set, x_old, parameter_set["tile_shape"])

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
        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
            self.assertEqual(tile_component.shape[:3], (1, 1, 1))
        self._assert_single_tile_continuity(
            parameter_set, tiled_particles.x[tiled_particles.active],
            tiled_particles.u[tiled_particles.active], current=J_tiles)


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


    def test_tiled_esirkepov_satisfies_discrete_continuity_at_conducting_particle_walls(self):
        walls = {"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC}
        parameter_set = self._build_parameter_values(
            Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1, particle_boundary_conditions=walls)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dx, dt = parameter_set["dx"], parameter_set["dt"]
        x_min = -0.5 * parameter_set["x_wind"]
        x_max = 0.5 * parameter_set["x_wind"]
        # two particles cross a wall during the step and reflect; one stays inside
        x_old = jnp.array([[x_min + 0.3 * dx, 0.0, 0.0],
                           [x_max - 0.2 * dx, 0.0, 0.0],
                           [0.1 * dx, 0.0, 0.0]])
        u = jnp.array([[-0.7 * dx / dt, 0.0, 0.0],
                       [0.6 * dx / dt, 0.0, 0.0],
                       [0.4 * dx / dt, 0.0, 0.0]])
        tiled_particles, species_config = self._particles_from_arrays(parameter_set, tile_shape, x_old, u)
        g = int(parameter_set["guard_cells"])
        n = tile_shape[0]
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
        new_particles = update_tiled_particle_positions(tiled_particles, species_config, dt)
        new_particles, overflow = refresh_tiled_particle_tiles(new_particles, static_parameters, dynamic_parameters)
        rho_new = compute_rho(new_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        # every owned C node, including both wall nodes g and g+n
        owned = (slice(None),) * 3 + (slice(g, g + n + 1), slice(g, -g), slice(g, -g))
        before = (slice(None),) * 3 + (slice(g - 1, g + n), slice(g, -g), slice(g, -g))
        drhodt = (rho_new[owned] - rho_old[owned]) / dt
        dJxdx = (J_tiles[0][owned] - J_tiles[0][before]) / dx
        continuity = drhodt + dJxdx
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(drhodt)) + jnp.max(jnp.abs(dJxdx)))
        self.assertGreater(float(jnp.max(jnp.abs(rho_new[owned][0, 0, 0, 0]))), 0.0)
        self.assertLessEqual(float(jnp.max(jnp.abs(continuity))), float(1.0e-12 * scale))


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
                    "particle_tile_nx": 2,
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
