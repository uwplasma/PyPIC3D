"""One-tile API contracts."""

from tests.support.pml_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    PMLConfigurationFixtures,
    PMLFDTDBehaviorFixtures,
    PMLInitializationFixtures,
    PML_WALLS,
    _base_parameter_values,
    _dynamic_parameters,
    _empty_config,
    _empty_global_fields,
    _initialize_pml_state,
    _load_pml,
    build_pml,
    ghost_cells,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    tempfile,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestPMLInitialization(PMLInitializationFixtures, unittest.TestCase):


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


class TestPMLFDTDBehavior(PMLFDTDBehaviorFixtures, unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
