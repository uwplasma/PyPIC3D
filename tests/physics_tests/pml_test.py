"""Single-device numerical tests."""

from tests.support.pml_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    PMLConfigurationFixtures,
    PMLFDTDBehaviorFixtures,
    PMLInitializationFixtures,
    PML_WALLS,
    SpeciesConfig,
    TiledParticles,
    _base_parameter_values,
    _dynamic_parameters,
    _empty_config,
    _empty_global_fields,
    _initialize_pml_state,
    _legacy_stretch_derivatives,
    _load_pml,
    assemble_yee_curl,
    build_pml,
    compute_energy,
    ghost_cells,
    initialize_simulation,
    initialize_tiled_pml_state,
    jax,
    jnp,
    kernel_parameters_from_values,
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
    tempfile,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestPMLConfiguration(PMLConfigurationFixtures, unittest.TestCase):
    def test_load_pml_from_toml_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "thickness": 2, "order": 3.0, "target_reflection": 1e-8}
            for wall in PML_WALLS
        ]

        active, pml_x, pml_y, pml_z, profiles = _load_pml(
            raw,
            _base_parameter_values(nx=8, ny=8, nz=8),
            {"C": 3.0},
        )
        sigma_x, sigma_y, sigma_z = profiles

        self.assertTrue(active)
        self.assertTrue(pml_x)
        self.assertTrue(pml_y)
        self.assertTrue(pml_z)
        self.assertEqual(sigma_x.shape, (10, 10, 10))
        self.assertTrue(jnp.any(sigma_x > 0.0))
        self.assertTrue(jnp.any(sigma_y > 0.0))
        self.assertTrue(jnp.any(sigma_z > 0.0))


    def test_load_pml_from_toml_rejects_invalid_duplicate_and_oversized_walls(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid PML wall"):
            _load_pml([{"wall": "x+", "thickness": 2}], parameter_set, {"C": 3.0})

        with self.assertRaisesRegex(ValueError, "Duplicate PML wall"):
            _load_pml(
                [{"wall": "+x", "thickness": 2}, {"wall": "+x", "thickness": 2}],
                parameter_set,
                {"C": 3.0},
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            _load_pml([{"wall": "+x", "thickness": 9}], parameter_set, {"C": 3.0})


    def test_build_pml_ramp_only_on_requested_side(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        pml_layers = (("+x", "x", 3, 2.0, 9.0),)

        sigma_x, sigma_y, sigma_z = build_pml(_dynamic_parameters(parameter_set), pml_layers)

        self.assertEqual(sigma_x.shape, (10, 3, 3))
        self.assertTrue(jnp.allclose(sigma_x[1:5, :, :], 0.0))
        self.assertTrue(jnp.all(sigma_x[-4:-1, :, :] > 0.0))
        self.assertTrue(float(sigma_x[-2, 1, 1]) > float(sigma_x[-4, 1, 1]))
        self.assertTrue(jnp.allclose(sigma_y, 0.0))
        self.assertTrue(jnp.allclose(sigma_z, 0.0))


    def test_build_pml_ramps_from_interior_interface_to_outer_wall(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        pml_layers = (
            ("-x", "x", 3, 2.0, 9.0),
            ("+x", "x", 3, 2.0, 9.0),
        )

        sigma_x, _, _ = build_pml(_dynamic_parameters(parameter_set), pml_layers)
        sigma_x = sigma_x[:, 1, 1]

        self.assertGreater(float(sigma_x[1]), float(sigma_x[2]))
        self.assertGreater(float(sigma_x[2]), float(sigma_x[3]))
        self.assertGreater(float(sigma_x[-2]), float(sigma_x[-3]))
        self.assertGreater(float(sigma_x[-3]), float(sigma_x[-4]))



class TestPMLInitialization(PMLInitializationFixtures, unittest.TestCase):
    def test_initialize_pml_state_uses_physical_interior_shape(self):
        parameter_set = _base_parameter_values(nx=8, ny=4, nz=2)
        parameter_set["pml"] = (False, False, False, False, build_pml(_dynamic_parameters(parameter_set), ()))

        pml_state = _initialize_pml_state(parameter_set)
        e_memory, b_memory, _ = pml_state

        self.assertEqual(len(e_memory), 6)
        self.assertEqual(len(b_memory), 6)
        for memory in e_memory:
            self.assertEqual(memory.shape, (8, 4, 2))
        for memory in b_memory:
            self.assertEqual(memory.shape, (8, 4, 2))


    def test_initialize_simulation_rejects_pml_for_electrostatic_solver(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "PML is only supported"):
                initialize_simulation(_empty_config(tmpdir, solver="electrostatic", pml=pml))


    def test_initialize_simulation_appends_pml_state_for_electrodynamic_yee(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir, pml=pml))

        fields = result[2]
        static_parameters = result[3]
        pml_state = fields[6]
        _, _, tiled_profiles = pml_state
        self.assertEqual(len(fields), 8)
        self.assertTrue(static_parameters.pml_active)
        self.assertIsNotNone(pml_state)
        self.assertTrue(jnp.any(tiled_profiles[0] > 0.0))
        self.assertTrue(jnp.allclose(tiled_profiles[1], 0.0))
        self.assertTrue(jnp.allclose(tiled_profiles[2], 0.0))
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))


    def test_initialize_simulation_uses_none_pml_state_without_pml_for_electrodynamic_yee(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir))

        fields = result[2]
        static_parameters = result[3]

        self.assertEqual(len(fields), 8)
        self.assertIsNone(fields[6])
        self.assertFalse(bool(fields[-1]))
        self.assertFalse(static_parameters.pml_active)



class TestPMLFDTDBehavior(PMLFDTDBehaviorFixtures, unittest.TestCase):
    def test_pml_curl_and_memory_match_legacy_trajectory(self):
        parameter_set = _base_parameter_values(nx=4, ny=4, nz=4)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [
                {"wall": "+x", "thickness": 2, "sigma_max": 3.0},
                {"wall": "+y", "thickness": 2, "sigma_max": 4.0},
                {"wall": "+z", "thickness": 2, "sigma_max": 5.0},
            ],
            parameter_set,
            dynamic_values,
        )
        tile_shape = (4, 4, 4)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        initial_state = initialize_tiled_pml_state(
            static_parameters,
            dynamic_parameters,
            parameter_set["pml"][-1],
            tile_shape,
        )

        g = int(static_parameters.guard_cells)
        profiles = tuple(profile[:, :, :, g:-g, g:-g, g:-g] for profile in initial_state[2])
        derivative_axes = (1, 2, 2, 0, 0, 1)

        for memory_name in ("zero", "random"):
            for side_name, stretch, memory_index, dt in (
                ("electric", stretch_tiled_pml_e_derivatives, 0, dynamic_parameters.dt),
                ("magnetic", stretch_tiled_pml_b_derivatives, 1, dynamic_parameters.dt / 2),
            ):
                with self.subTest(memory=memory_name, side=side_name):
                    if memory_name == "zero":
                        state = initial_state
                    else:
                        keys = jax.random.split(jax.random.key(51), 12)
                        e_memory = tuple(
                            jax.random.normal(key, memory.shape, dtype=memory.dtype)
                            for key, memory in zip(keys[:6], initial_state[0])
                        )
                        b_memory = tuple(
                            jax.random.normal(key, memory.shape, dtype=memory.dtype)
                            for key, memory in zip(keys[6:], initial_state[1])
                        )
                        state = (e_memory, b_memory, initial_state[2])

                    reference_state = state
                    for step in range(4):
                        keys = jax.random.split(jax.random.key(100 + step), 6)
                        derivatives = tuple(
                            jax.random.normal(key, memory.shape, dtype=memory.dtype)
                            for key, memory in zip(keys, state[memory_index])
                        )

                        stretched, state = stretch(
                            derivatives,
                            static_parameters,
                            dynamic_parameters,
                            state,
                        )
                        reference_stretched, reference_memory = _legacy_stretch_derivatives(
                            derivatives,
                            reference_state[memory_index],
                            profiles,
                            dt,
                            derivative_axes,
                        )
                        if memory_index == 0:
                            reference_state = (
                                reference_memory,
                                reference_state[1],
                                reference_state[2],
                            )
                        else:
                            reference_state = (
                                reference_state[0],
                                reference_memory,
                                reference_state[2],
                            )

                        for actual, expected in zip(
                            assemble_yee_curl(stretched),
                            assemble_yee_curl(reference_stretched),
                        ):
                            self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))
                        for actual, expected in zip(state[0] + state[1], reference_state[0] + reference_state[1]):
                            self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))


    def test_magnetic_pml_memory_uses_b_half_timestep(self):
        parameter_set = _base_parameter_values(nx=4, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "sigma_max": 4.0}],
            parameter_set,
            dynamic_values,
        )
        tile_shape = (4, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        pml_state = initialize_tiled_pml_state(
            static_parameters,
            dynamic_parameters,
            parameter_set["pml"][-1],
            tile_shape,
        )

        _, b_memory, tiled_profiles = pml_state
        derivatives = tuple(jnp.ones_like(memory) for memory in b_memory)
        _, pml_state = stretch_tiled_pml_b_derivatives(
            derivatives,
            static_parameters,
            dynamic_parameters,
            pml_state,
        )

        g = int(static_parameters.guard_cells)
        sigma_x = tiled_profiles[0][:, :, :, g:-g, g:-g, g:-g]
        expected_x_memory = jnp.exp(-sigma_x * dynamic_parameters.dt / 2) - 1.0
        b_memory = pml_state[1]

        self.assertTrue(jnp.allclose(b_memory[3], expected_x_memory))
        self.assertTrue(jnp.allclose(b_memory[4], expected_x_memory))


    def test_no_pml_state_returns_field_and_none_state(self):
        parameter_set = _base_parameter_values(nx=4, ny=4, nz=4)
        dynamic_values = {"C": 2.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 4, 4)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        E, B, J = _empty_global_fields(parameter_set)
        B = (B[0], B[1], B[2].at[1:-1, 1:-1, 1:-1].set(1.0))

        E_tiles = tile_vector_field(E, parameter_set, tile_shape)
        B_tiles = tile_vector_field(B, parameter_set, tile_shape)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E_after, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_after, pml_state = update_B(E_after, B_tiles, static_parameters, dynamic_parameters, pml_state)

        self.assertEqual(len(E_after), 3)
        self.assertEqual(len(B_after), 3)
        self.assertIsNone(pml_state)


    def test_tiled_pml_absorbs_field_energy_in_particle_free_1d_wave(self):
        parameter_set = _base_parameter_values(nx=40, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
                {"wall": "+x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
            ],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (40, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        tiled_pml_state = initialize_tiled_pml_state(
            static_parameters,
            dynamic_parameters,
            parameter_set["pml"][-1],
            tile_shape,
        )
        E, B, J = _empty_global_fields(parameter_set)

        x = parameter_set["grids"]["vertex"][0][1:-1]
        pulse = jnp.exp(-((x + 0.30) / 0.04) ** 2)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(pulse)
        Bz = Bz.at[1:-1, 1, 1].set(pulse)
        E_tiles = tile_vector_field((Ex, Ey, Ez), parameter_set, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), parameter_set, tile_shape)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape)

        empty_particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        )
        empty_species_config = SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
        )

        initial_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        def step(E_tiles, B_tiles, tiled_pml_state):
            B_tiles, tiled_pml_state = update_B(
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            E_tiles, tiled_pml_state = update_E(
                E_tiles,
                B_tiles,
                J_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            B_tiles, tiled_pml_state = update_B(
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            return E_tiles, B_tiles, tiled_pml_state

        step = jax.jit(step)
        for _ in range(60):
            E_tiles, B_tiles, tiled_pml_state = step(E_tiles, B_tiles, tiled_pml_state)

        final_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        self.assertTrue(jnp.isfinite(final_energy))
        self.assertLess(float(final_energy), 0.65 * float(initial_energy))


    def test_tiled_pml_step_uses_shared_guard_current_without_changing_timing(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 2.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "order": 3.0, "sigma_max": 4.0}],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        E, B, J = _empty_global_fields(parameter_set)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(jnp.linspace(0.0, 0.3, parameter_set["Nx"]))
        Bz = Bz.at[1:-1, 1, 1].set(jnp.linspace(0.1, -0.2, parameter_set["Nx"]))
        E_tiles = tile_vector_field((Ex, Ey, Ez), parameter_set, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), parameter_set, tile_shape)

        g = int(parameter_set["guard_cells"])
        J_tiles = tile_vector_field(J, parameter_set, tile_shape, num_guard_cells=g)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 1:-1, 1:-1, 1:-1].set(0.25)
        J_tiles = (Jx, Jy, Jz)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        pml_state = initialize_tiled_pml_state(static_parameters, dynamic_parameters, parameter_set["pml"][-1], tile_shape)
        E_after, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state)
        B_after, pml_state = update_B(E_after, B_tiles, static_parameters, dynamic_parameters, pml_state)

        for component in E_after + B_after:
            self.assertTrue(jnp.all(jnp.isfinite(component)))
        for memory in pml_state[0] + pml_state[1]:
            self.assertTrue(jnp.all(jnp.isfinite(memory)))



if __name__ == "__main__":
    unittest.main()
