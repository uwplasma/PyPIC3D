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
