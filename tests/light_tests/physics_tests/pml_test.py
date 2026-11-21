"""One-tile numerical and physical checks."""

from tests.support.pml_fixtures import (
    BC_CONDUCTING,
    PMLConfigurationFixtures,
    PMLFDTDBehaviorFixtures,
    SpeciesConfig,
    TiledParticles,
    _base_parameter_values,
    _dynamic_parameters,
    _empty_global_fields,
    _legacy_stretch_derivatives,
    _load_pml,
    assemble_yee_curl,
    build_pml,
    compute_energy,
    ghost_cells,
    initialize_tiled_pml_state,
    jax,
    jnp,
    kernel_parameters_from_values,
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestPMLConfiguration(PMLConfigurationFixtures, unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
