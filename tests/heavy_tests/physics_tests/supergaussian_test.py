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


class TestSupergaussianFDTDBehavior(unittest.TestCase):
    def test_tiled_supergaussian_absorbs_field_energy_in_particle_free_1d_wave(self):
        parameter_set = _base_parameter_values(nx=40, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["supergaussian"] = _load_supergaussian(
            [
                {"wall": "-x", "width": 8, "order": 4.0, "sigma_max": 60.0},
                {"wall": "+x", "width": 8, "order": 4.0, "sigma_max": 60.0},
            ],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (40, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J = _empty_global_fields(parameter_set)

        x = parameter_set["grids"]["vertex"][0][1:-1]
        pulse = jnp.exp(-((x + 0.30) / 0.04) ** 2)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(pulse)
        Bz = Bz.at[1:-1, 1, 1].set(pulse)
        E_tiles = tile_vector_field((Ex, Ey, Ez), parameter_set, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), parameter_set, tile_shape)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape)

        empty_particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        )
        empty_species_config = SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
        )

        initial_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )

        def step(E_tiles, B_tiles):
            B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, None)
            E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state)
            B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)
            return E_tiles, B_tiles

        step = jax.jit(step)
        for _ in range(60):
            E_tiles, B_tiles = step(E_tiles, B_tiles)

        final_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        self.assertTrue(jnp.isfinite(final_energy))
        self.assertLess(float(final_energy), 0.65 * float(initial_energy))


if __name__ == "__main__":
    unittest.main()
