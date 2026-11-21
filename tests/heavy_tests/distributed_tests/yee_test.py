from tests.support.distributed.yee import YeeTiledFixtures
"""Focused cross-device regression tests."""

from tests.support.yee_fixtures import (
    SimpleNamespace,
    assemble_tiled_vector_field,
    ghost_cells,
    build_tiled_yee_grids,
    build_yee_grid,
    jnp,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestYeeTiled(YeeTiledFixtures, unittest.TestCase):
    def test_tile_vector_field_assembles_to_original_ghost_celled_field(self):
        parameter_set = self._build_parameter_values()
        tile_shape = (4, 3, 2)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)

        E_tiles = tile_vector_field(E, parameter_set, tile_shape)
        E_assembled = assemble_tiled_vector_field(E_tiles, parameter_set, tile_shape)

        for original, assembled in zip(E, E_assembled):
            self.assertTrue(jnp.allclose(assembled, original, rtol=1.0e-12, atol=1.0e-12))


    def test_update_E_matches_single_tile_yee_update(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        J = self._deterministic_vector_field(parameter_set, scale=0.05)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        E_reference, _ = self._reference_update_E(E, B, J, parameter_set, dynamic_values)
        E_tiled, pml_state = update_E(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            tile_vector_field(J, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        E_from_tiles = assemble_tiled_vector_field(E_tiled, parameter_set, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


    def test_update_E_matches_single_tile_yee_update_with_conducting_boundaries(self):
        parameter_set = self._conducting_parameters()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        J = self._deterministic_vector_field(parameter_set, scale=0.05)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        E_reference, _ = self._reference_update_E(E, B, J, parameter_set, dynamic_values)
        E_tiled, pml_state = update_E(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            tile_vector_field(J, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        E_from_tiles = assemble_tiled_vector_field(E_tiled, parameter_set, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


    def test_update_B_matches_single_tile_yee_update(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        B_reference, _ = self._reference_update_B(E, B, parameter_set, dynamic_values)
        B_tiled, pml_state = update_B(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        B_from_tiles = assemble_tiled_vector_field(B_tiled, parameter_set, tile_shape)

        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


    def test_update_B_matches_single_tile_yee_update_with_conducting_boundaries(self):
        parameter_set = self._conducting_parameters()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tile_metadata(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        static_parameters, dynamic_parameters = self._split_parameters(parameter_set, dynamic_values)

        B_reference, _ = self._reference_update_B(E, B, parameter_set, dynamic_values)
        B_tiled, pml_state = update_B(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        B_from_tiles = assemble_tiled_vector_field(B_tiled, parameter_set, tile_shape)

        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


    def test_tiled_electrodynamic_step_matches_single_tile_yee_sequence(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
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
        self.assertIsNone(pml_state)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, parameter_set, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, parameter_set, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


    def test_tiled_electrodynamic_step_matches_single_tile_yee_sequence_with_conducting_boundaries(self):
        parameter_set = self._conducting_parameters()
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 3, 2)
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
        self.assertIsNone(pml_state)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, parameter_set, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, parameter_set, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
