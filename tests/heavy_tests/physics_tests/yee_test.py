"""One-tile numerical and physical checks."""

from tests.support.yee_fixtures import (
    YeeTiledFixtures,
    _field_dot,
    ghost_cells,
    jnp,
    unittest,
    yee_curl_b_to_e,
    yee_curl_e_to_b,
    yee_derivatives_b_to_e,
    yee_derivatives_e_to_b,
)


class TestYeeTiled(YeeTiledFixtures, unittest.TestCase):
    def test_scalar_yee_derivatives_satisfy_adjoint_identity(self):
        forward_to_backward = (5, 2, 1, 4, 3, 0)
        source_components = (2, 1, 0, 2, 1, 0)
        target_components = (0, 0, 1, 1, 2, 2)

        for boundary_name, parameter_builder in (
            ("periodic", self._build_parameter_values),
            ("conducting", self._conducting_parameters),
        ):
            tile_shape = (8, 6, 4)
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
            tile_shape = (8, 6, 4)
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


if __name__ == "__main__":
    unittest.main()
