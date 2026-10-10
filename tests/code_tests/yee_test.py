"""Single-device numerical tests."""

from tests.support.yee_fixtures import (
    YeeTiledFixtures,
    _field_dot,
    assemble_tiled_vector_field,
    ghost_cells,
    jnp,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
    yee_curl_b_to_e,
    yee_curl_e_to_b,
    yee_derivatives_b_to_e,
    yee_derivatives_e_to_b,
)


class TestYeeTiled(YeeTiledFixtures, unittest.TestCase):
    def test_eager_and_compiled_public_updates_agree(self):
        from tests.support.compiled_yee import eager
        parameters = self._build_parameter_values()
        shape = (8, 6, 4)
        parameters = self._with_tile_metadata(parameters, shape)
        static, dynamic = self._split_parameters(parameters, {"C": 1., "eps": 1.})
        E = tile_vector_field(self._deterministic_vector_field(parameters, 1.), parameters, shape)
        B = tuple(component * .2 for component in E)
        J = tuple(component * .05 for component in E)
        eager_E, _ = eager.update_E(E, B, J, static, dynamic)
        compiled_E, _ = update_E(E, B, J, static, dynamic)
        eager_B, _ = eager.update_B(E, B, static, dynamic)
        compiled_B, _ = update_B(E, B, static, dynamic)
        for actual, expected in zip(compiled_E + compiled_B, eager_E + eager_B):
            self.assertTrue(jnp.allclose(actual, expected, rtol=1.e-12, atol=1.e-12))

    def test_scalar_yee_derivatives_satisfy_adjoint_identity(self):
        forward_to_backward = (5, 2, 1, 4, 3, 0)
        source_components = (2, 1, 0, 2, 1, 0)
        target_components = (0, 0, 1, 1, 2, 2)

        for boundary_name, parameter_builder in (
            ("periodic", self._build_parameter_values),
            ("conducting", self._conducting_parameters),
        ):
            for tile_shape in ((8, 6, 4),):
                with self.subTest(boundary=boundary_name, tile_shape=tile_shape):
                    parameter_set = self._with_tile_metadata(parameter_builder(), tile_shape)
                    static_parameters, dynamic_parameters = self._split_parameters(parameter_set, {})
                    E_tiles = self._random_tiled_vector_field(parameter_set, tile_shape, seed=14)
                    B_tiles = self._random_tiled_vector_field(parameter_set, tile_shape, seed=15)
                    # The upper C plane is now a physical endpoint. Random
                    # fields must satisfy PEC there before applying adjoint curls.
                    E_tiles = ghost_cells.apply_tiled_pec_boundary(E_tiles, static_parameters)
                    forward = yee_derivatives_e_to_b(E_tiles, static_parameters, dynamic_parameters)
                    backward = yee_derivatives_b_to_e(
                        B_tiles,
                        static_parameters,
                        dynamic_parameters,
                    )

                    g = int(static_parameters.guard_cells)
                    E_active = tuple(component[:, :, :, g:-g, g:-g, g:-g] for component in E_tiles)
                    B_active = tuple(component[:, :, :, g:-g, g:-g, g:-g] for component in B_tiles)
                    for forward_index, backward_index in enumerate(forward_to_backward):
                        lhs = jnp.vdot(
                            forward[forward_index],
                            B_active[target_components[forward_index]],
                        )
                        rhs = -jnp.vdot(
                            E_active[source_components[forward_index]],
                            backward[backward_index],
                        )
                        self.assertTrue(jnp.allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12))


    def test_periodic_and_conducting_curls_satisfy_discrete_integration_by_parts(self):
        for boundary_name, parameter_set in (
            ("periodic", self._build_parameter_values()),
            ("conducting", self._conducting_parameters()),
        ):
            for tile_shape in ((8, 6, 4),):
                with self.subTest(boundary=boundary_name, tile_shape=tile_shape):
                    tiled_parameters = self._with_tile_metadata(parameter_set, tile_shape)
                    static_parameters, dynamic_parameters = self._split_parameters(tiled_parameters, {})
                    E_tiles = self._random_tiled_vector_field(tiled_parameters, tile_shape, seed=31)
                    B_tiles = self._random_tiled_vector_field(tiled_parameters, tile_shape, seed=32)
                    # The upper C plane is now a physical endpoint. Random
                    # fields must satisfy PEC there before applying adjoint curls.
                    E_tiles = ghost_cells.apply_tiled_pec_boundary(E_tiles, static_parameters)

                    curl_E = yee_curl_e_to_b(E_tiles, static_parameters, dynamic_parameters)
                    curl_B = yee_curl_b_to_e(
                        B_tiles,
                        static_parameters,
                        dynamic_parameters,
                    )

                    g = int(static_parameters.guard_cells)
                    E_active = tuple(component[:, :, :, g:-g, g:-g, g:-g] for component in E_tiles)
                    B_active = tuple(component[:, :, :, g:-g, g:-g, g:-g] for component in B_tiles)
                    lhs = _field_dot(curl_E, B_active)
                    rhs = _field_dot(E_active, curl_B)

                    self.assertTrue(jnp.allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12))


    def test_conducting_update_clamps_only_tangential_e_components(self):
        parameter_set = self._conducting_parameters()
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, {"alpha": 1.0})
        E = tuple(jnp.ones((10, 8, 6), dtype=jnp.float64) * value for value in (2.0, 3.0, 5.0))
        zero = tuple(jnp.zeros((10, 8, 6), dtype=jnp.float64) for _ in range(3))

        E_tiles, pml_state = update_E(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(zero, parameter_set, tile_shape),
            tile_vector_field(zero, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )

        self.assertIsNone(pml_state)
        g = int(static_parameters.guard_cells)
        Ex, Ey, Ez = E_tiles

        self.assertTrue(jnp.all(Ey[0, :, :, g, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, :, :, g, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[:, 0, :, :, g, :] == 0.0))
        self.assertTrue(jnp.all(Ez[:, 0, :, :, g, :] == 0.0))
        self.assertTrue(jnp.all(Ex[:, :, 0, :, :, g] == 0.0))
        self.assertTrue(jnp.all(Ey[:, :, 0, :, :, g] == 0.0))

        self.assertEqual(float(Ex[0, 0, 0, g, g + 1, g + 1]), 2.0)
        self.assertEqual(float(Ey[0, 0, 0, g + 1, g, g + 1]), 3.0)
        self.assertEqual(float(Ez[0, 0, 0, g + 1, g + 1, g]), 5.0)


    def test_field_updates_do_not_use_coupling_filter_alpha(self):
        parameter_set = self._build_parameter_values()
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = tile_vector_field(self._deterministic_vector_field(parameter_set, scale=1.0), parameter_set, tile_shape)
        B = tile_vector_field(self._deterministic_vector_field(parameter_set, scale=0.2), parameter_set, tile_shape)
        J = tile_vector_field(self._deterministic_vector_field(parameter_set, scale=0.05), parameter_set, tile_shape)

        static_06, dynamic_06 = self._split_parameters(
            parameter_set,
            {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 0.6},
        )
        static_10, dynamic_10 = self._split_parameters(
            parameter_set,
            {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0},
        )

        E_06, _ = update_E(E, B, J, static_06, dynamic_06)
        E_10, _ = update_E(E, B, J, static_10, dynamic_10)
        B_06, _ = update_B(E, B, static_06, dynamic_06)
        B_10, _ = update_B(E, B, static_10, dynamic_10)

        for field_06, field_10 in zip(E_06 + B_06, E_10 + B_10):
            self.assertTrue(jnp.array_equal(field_06, field_10))


    def test_update_E_is_the_public_tiled_update(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        J = self._deterministic_vector_field(parameter_set, scale=0.05)
        E_tiles = tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=2)
        B_tiles = tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=2)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape, num_guard_cells=2)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        E_public, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertEqual(E_public[0].ndim, 6)
        self.assertEqual(E_public[0].shape[:3], E_tiles[0].shape[:3])


    def test_update_B_is_the_public_tiled_update(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        E_tiles = tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=2)
        B_tiles = tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=2)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        B_public, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertEqual(B_public[0].ndim, 6)
        self.assertEqual(B_public[0].shape[:3], B_tiles[0].shape[:3])


    def test_update_E_uses_constant_boundary_without_conducting_tangential_zeroing(self):
        parameter_set = self._constant_x_parameters()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = tuple(jnp.ones((10, 8, 6), dtype=float) * value for value in (2.0, 3.0, 5.0))
        B = tuple(jnp.zeros((10, 8, 6), dtype=float) for _ in range(3))
        J = tuple(jnp.zeros((10, 8, 6), dtype=float) for _ in range(3))
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        E_tiled, pml_state = update_E(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            tile_vector_field(J, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )

        self.assertIsNone(pml_state)
        self._assert_x_ghosts_are_constant(E_tiled, int(static_parameters.guard_cells))
        self.assertTrue(jnp.allclose(E_tiled[1][:, :, :, int(static_parameters.guard_cells), :, :], 3.0))
        self.assertTrue(jnp.allclose(E_tiled[2][:, :, :, int(static_parameters.guard_cells), :, :], 5.0))


    def test_update_B_uses_constant_boundary(self):
        parameter_set = self._constant_x_parameters()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = tuple(jnp.zeros((10, 8, 6), dtype=float) for _ in range(3))
        B = tuple(jnp.ones((10, 8, 6), dtype=float) * value for value in (7.0, 11.0, 13.0))
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        B_tiled, pml_state = update_B(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )

        self.assertIsNone(pml_state)
        self._assert_x_ghosts_are_constant(B_tiled, int(static_parameters.guard_cells))


    def test_tiled_evolve_updates_fields_without_pushing_particles(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        J = self._deterministic_vector_field(parameter_set, scale=0.05)

        E_reference, B_reference, _ = self._reference_yee_step(E, B, J, parameter_set, dynamic_values)
        E_tiles = tile_vector_field(E, parameter_set, tile_shape)
        B_tiles = tile_vector_field(B, parameter_set, tile_shape)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)
        E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, parameter_set, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, parameter_set, tile_shape)

        self.assertIsNone(pml_state)
        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for original, after in zip(tile_vector_field(J, parameter_set, tile_shape), J_tiles):
            self.assertTrue(jnp.allclose(after, original, rtol=1.0e-12, atol=1.0e-12))



if __name__ == "__main__":
    unittest.main()
