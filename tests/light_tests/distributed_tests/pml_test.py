"""Focused cross-device regression tests."""

from tests.support.pml_fixtures import (
    BC_CONDUCTING,
    PMLFDTDBehaviorFixtures,
    PMLInitializationFixtures,
    _base_parameter_values,
    _empty_config,
    _empty_global_fields,
    _initialize_tiled_pml_state,
    _load_pml,
    _tile_pml_profiles,
    _update_ghost_cells_from_parameters,
    assemble_tiled_vector_field,
    ghost_cells,
    initialize_simulation,
    initialize_tiled_pml_state,
    jnp,
    kernel_parameters_from_values,
    tempfile,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestPMLInitialization(PMLInitializationFixtures, unittest.TestCase):
    def test_initialize_tiled_pml_state_uses_tile_local_interior_shape(self):
        parameter_set = _base_parameter_values(nx=8, ny=4, nz=2)
        parameter_set["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "sigma_max": 3.0}],
            parameter_set,
            {"C": 1.0},
        )
        tile_shape = (4, 2, 1)

        pml_state = _initialize_tiled_pml_state(parameter_set, tile_shape, {"C": 1.0})
        e_memory, b_memory, tiled_profiles = pml_state

        self.assertEqual(len(e_memory), 6)
        self.assertEqual(len(b_memory), 6)
        for memory in e_memory:
            self.assertEqual(memory.shape, (2, 2, 2, 4, 2, 1))
        for memory in b_memory:
            self.assertEqual(memory.shape, (2, 2, 2, 4, 2, 1))
        for profile in tiled_profiles:
            self.assertEqual(profile.shape, (2, 2, 2, 8, 6, 5))


    def test_tile_pml_profiles_matches_global_profiles_on_tile_interiors(self):
        parameter_set = _base_parameter_values(nx=8, ny=4, nz=2)
        parameter_set["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 2, "sigma_max": 3.0},
                {"wall": "+x", "thickness": 2, "sigma_max": 3.0},
            ],
            parameter_set,
            {"C": 1.0},
        )
        tile_shape = (4, 2, 1)

        tiled_profiles = _tile_pml_profiles(parameter_set, tile_shape, {"C": 1.0})
        global_profiles = parameter_set["pml"][-1]
        assembled_profiles = assemble_tiled_vector_field(tiled_profiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))

        for assembled, reference in zip(assembled_profiles, global_profiles):
            self.assertTrue(
                jnp.allclose(
                    assembled[1:-1, 1:-1, 1:-1],
                    reference[1:-1, 1:-1, 1:-1],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )


if __name__ == "__main__":
    unittest.main()
