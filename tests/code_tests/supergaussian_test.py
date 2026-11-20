from tests.support.supergaussian_runtime_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    SUPERGAUSSIAN_WALLS,
    _base_parameter_values,
    _empty_config,
    _empty_global_fields,
    _load_supergaussian,
    apply_tiled_supergaussian_absorber,
    assemble_tiled_vector_field,
    ghost_cells,
    initialize_simulation,
    jnp,
    kernel_parameters_from_values,
    tempfile,
    tile_vector_field,
    unittest,
)


class TestSupergaussianConfiguration(unittest.TestCase):
    def test_load_supergaussian_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "width": 2, "order": 4.0, "target_reflection": 1.0e-8}
            for wall in SUPERGAUSSIAN_WALLS
        ]

        active, sg_x, sg_y, sg_z, layers = _load_supergaussian(raw, _base_parameter_values(8, 8, 8), {"C": 3.0})

        self.assertTrue(active)
        self.assertTrue(sg_x)
        self.assertTrue(sg_y)
        self.assertTrue(sg_z)
        self.assertEqual(len(layers), 6)

    def test_load_supergaussian_rejects_invalid_duplicate_and_oversized_walls(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid supergaussian wall"):
            _load_supergaussian([{"wall": "x+", "width": 2}], parameter_set)

        with self.assertRaisesRegex(ValueError, "Duplicate supergaussian wall"):
            _load_supergaussian(
                [{"wall": "+x", "width": 2}, {"wall": "+x", "width": 2}],
                parameter_set,
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            _load_supergaussian([{"wall": "+x", "width": 9}], parameter_set)


class TestSupergaussianInitialization(unittest.TestCase):
    def test_initialize_simulation_rejects_supergaussian_for_electrostatic_solver(self):
        supergaussian = [{"wall": "+x", "width": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "not supported for the electrostatic solver"):
                initialize_simulation(_empty_config(tmpdir, solver="electrostatic", supergaussian=supergaussian))

    def test_initialize_simulation_stores_static_supergaussian_layers(self):
        supergaussian = [{"wall": "+x", "width": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir, supergaussian=supergaussian))

        fields = result[2]
        static_parameters = result[3]
        self.assertEqual(len(fields), 8)
        self.assertTrue(static_parameters.supergaussian_active)
        self.assertEqual(len(static_parameters.supergaussian_layers), 1)
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))

    def test_initialize_simulation_allows_static_metric_supergaussian_with_constant_radial_bc(self):
        supergaussian = [
            {"wall": "-x", "width": 2, "sigma_max": 1.0},
            {"wall": "+x", "width": 2, "sigma_max": 1.0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _empty_config(tmpdir, solver="static_metric", supergaussian=supergaussian)
            config["simulation_parameters"].update(
                {
                    "metric": "flat_spherical",
                    "particle_pusher": "hybrid_boris_geodesic",
                    "current_calculation": "GR_direct_deposition",
                    "x_bc": "constant",
                    "y_bc": "periodic",
                    "z_bc": "periodic",
                    "x_min": 1.0,
                    "x_max": 2.0,
                    "y_min": 0.17,
                    "y_max": 1.17,
                }
            )

            result = initialize_simulation(config)

        fields = result[2]
        static_parameters = result[3]
        self.assertEqual(len(fields), 9)
        self.assertTrue(static_parameters.supergaussian_active)
        self.assertEqual(len(static_parameters.supergaussian_layers), 2)
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONSTANT, BC_PERIODIC, BC_PERIODIC))


class TestSupergaussianFDTDBehavior(unittest.TestCase):
    def test_tiled_supergaussian_absorber_damps_E_and_B_but_not_J(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        parameter_set["supergaussian"] = _load_supergaussian(
            [{"wall": "+x", "width": 3, "order": 2.0, "sigma_max": 20.0}],
            parameter_set,
        )
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J = _empty_global_fields(parameter_set)
        E_tiles = tile_vector_field((E[0], E[1].at[1:-1, 1, 1].set(1.0), E[2]), parameter_set, tile_shape)
        B_tiles = tile_vector_field((B[0], B[1], B[2].at[1:-1, 1, 1].set(1.0)), parameter_set, tile_shape)
        J_tiles = tile_vector_field((J[0], J[1].at[1:-1, 1, 1].set(1.0), J[2]), parameter_set, tile_shape)

        E_after = apply_tiled_supergaussian_absorber(E_tiles, static_parameters, dynamic_parameters, dynamic_parameters.dt)
        B_after = apply_tiled_supergaussian_absorber(B_tiles, static_parameters, dynamic_parameters, dynamic_parameters.dt)

        E_global = assemble_tiled_vector_field(E_after, parameter_set, tile_shape)
        B_global = assemble_tiled_vector_field(B_after, parameter_set, tile_shape)
        J_global = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape)
        self.assertLess(float(E_global[1][-2, 1, 1]), 1.0)
        self.assertLess(float(B_global[2][-2, 1, 1]), 1.0)
        self.assertTrue(jnp.allclose(J_global[1][1:-1, 1, 1], 1.0))


if __name__ == "__main__":
    unittest.main()
