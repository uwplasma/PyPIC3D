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


    def test_initialize_simulation_uses_tiled_pml_state_for_electrodynamic_yee(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _empty_config(tmpdir, solver="electrodynamic_yee", pml=pml)
            config["simulation_parameters"].update(
                {
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                }
            )
            result = initialize_simulation(config)

        fields = result[2]
        static_parameters = result[3]
        pml_state = fields[6]
        overflow = fields[-1]
        e_memory, b_memory, tiled_profiles = pml_state

        self.assertEqual(len(fields), 8)
        self.assertFalse(bool(overflow))
        self.assertEqual(tuple(static_parameters.tile_shape), (2, 1, 1))
        self.assertEqual(e_memory[0].shape, (4, 1, 1, 2, 1, 1))
        self.assertEqual(b_memory[0].shape, (4, 1, 1, 2, 1, 1))
        self.assertEqual(int(static_parameters.guard_cells), 2)
        self.assertEqual(tiled_profiles[0].shape, (4, 1, 1, 6, 5, 5))


class TestPMLFDTDBehavior(PMLFDTDBehaviorFixtures, unittest.TestCase):
    def test_tiled_pml_matches_single_tile_pml_for_one_yee_step(self):
        parameter_set = _base_parameter_values(nx=8, ny=4, nz=2)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 2, "sigma_max": 4.0},
                {"wall": "+x", "thickness": 2, "sigma_max": 4.0},
            ],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (4, 2, 1)
        E, B, J = _empty_global_fields(parameter_set)

        x = parameter_set["grids"]["vertex"][0][1:-1]
        y = parameter_set["grids"]["vertex"][1][1:-1]
        z = parameter_set["grids"]["vertex"][2][1:-1]
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Jx, Jy, Jz = J
        Ey = Ey.at[1:-1, 1:-1, 1:-1].set(jnp.sin(2.0 * jnp.pi * X) + 0.1 * Y)
        Ez = Ez.at[1:-1, 1:-1, 1:-1].set(0.2 * X - 0.3 * Z)
        By = By.at[1:-1, 1:-1, 1:-1].set(0.4 * X + 0.2 * Z)
        Bz = Bz.at[1:-1, 1:-1, 1:-1].set(jnp.cos(2.0 * jnp.pi * X) - 0.1 * Y)
        Jx = Jx.at[1:-1, 1:-1, 1:-1].set(0.05 * X)
        Jy = Jy.at[1:-1, 1:-1, 1:-1].set(-0.02 * Y)
        Jz = Jz.at[1:-1, 1:-1, 1:-1].set(0.03 * Z)
        E = (Ex, Ey, Ez)
        B = (Bx, By, Bz)
        J = (Jx, Jy, Jz)
        E = tuple(_update_ghost_cells_from_parameters(component, parameter_set) for component in E)
        B = tuple(_update_ghost_cells_from_parameters(component, parameter_set) for component in B)

        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["tile_shape"] = reference_tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        reference_static_parameters, reference_dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        reference_pml_state = initialize_tiled_pml_state(
            reference_static_parameters,
            reference_dynamic_parameters,
            parameter_set["pml"][-1],
            reference_tile_shape,
        )
        reference_E, reference_pml_state = update_E(
            tile_vector_field(E, parameter_set, reference_tile_shape),
            tile_vector_field(B, parameter_set, reference_tile_shape),
            tile_vector_field(J, parameter_set, reference_tile_shape),
            reference_static_parameters,
            reference_dynamic_parameters,
            reference_pml_state,
        )
        reference_B, reference_pml_state = update_B(
            reference_E,
            tile_vector_field(B, parameter_set, reference_tile_shape),
            reference_static_parameters,
            reference_dynamic_parameters,
            reference_pml_state,
        )
        E_reference = assemble_tiled_vector_field(
            reference_E,
            parameter_set,
            reference_tile_shape,
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        B_reference = assemble_tiled_vector_field(
            reference_B,
            parameter_set,
            reference_tile_shape,
            num_guard_cells=int(parameter_set["guard_cells"]),
        )

        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((2, 2, 2))
        tiled_static_parameters, tiled_dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        tiled_pml_state = initialize_tiled_pml_state(
            tiled_static_parameters,
            tiled_dynamic_parameters,
            parameter_set["pml"][-1],
            tile_shape,
        )
        E_tiles, tiled_pml_state = update_E(
            tile_vector_field(E, parameter_set, tile_shape),
            tile_vector_field(B, parameter_set, tile_shape),
            tile_vector_field(J, parameter_set, tile_shape),
            tiled_static_parameters,
            tiled_dynamic_parameters,
            tiled_pml_state,
        )
        B_tiles, tiled_pml_state = update_B(
            E_tiles,
            tile_vector_field(B, parameter_set, tile_shape),
            tiled_static_parameters,
            tiled_dynamic_parameters,
            tiled_pml_state,
        )

        E_tiled = assemble_tiled_vector_field(E_tiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))
        B_tiled = assemble_tiled_vector_field(B_tiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        for reference, tiled in zip(E_reference, E_tiled):
            self.assertTrue(jnp.allclose(tiled[interior], reference[interior], rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_tiled):
            self.assertTrue(jnp.allclose(tiled[interior], reference[interior], rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
