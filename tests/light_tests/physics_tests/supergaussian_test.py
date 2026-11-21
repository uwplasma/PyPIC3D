from tests.support.supergaussian_runtime_fixtures import (
    BC_CONDUCTING,
    SpeciesConfig,
    TiledParticles,
    _base_parameter_values,
    _empty_global_fields,
    _load_supergaussian,
    assemble_tiled_vector_field,
    build_supergaussian_envelope,
    compute_energy,
    ghost_cells,
    jax,
    jnp,
    kernel_parameters_from_values,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestSupergaussianConfiguration(unittest.TestCase):
    def test_build_envelope_damps_only_requested_wall(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        parameter_set["supergaussian"] = _load_supergaussian(
            [{"wall": "+x", "width": 3, "order": 2.0, "sigma_max": 9.0}],
            parameter_set,
        )
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)

        envelope = build_supergaussian_envelope(static_parameters, dynamic_parameters, dynamic_parameters.dt)
        assembled = assemble_tiled_vector_field((envelope, envelope, envelope), parameter_set, tile_shape)[0]
        line = assembled[1:-1, 1, 1]

        self.assertTrue(jnp.allclose(line[:5], 1.0))
        self.assertLess(float(line[-1]), float(line[-2]))
        self.assertLess(float(line[-2]), float(line[-3]))
        self.assertGreater(float(line[-3]), 0.0)


if __name__ == "__main__":
    unittest.main()
