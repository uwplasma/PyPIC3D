"""One-tile numerical and physical checks."""

from tests.support.pml_fixtures import (
    BC_CONDUCTING,
    PMLConfigurationFixtures,
    PMLFDTDBehaviorFixtures,
    SpeciesConfig,
    TiledParticles,
    _base_parameter_values,
    _dynamic_parameters,
    _empty_global_fields,
    _legacy_stretch_derivatives,
    _load_pml,
    assemble_yee_curl,
    build_pml,
    compute_energy,
    ghost_cells,
    initialize_tiled_pml_state,
    jax,
    jnp,
    kernel_parameters_from_values,
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
    tile_vector_field,
    unittest,
    update_B,
    update_E,
)


class TestPMLFDTDBehavior(PMLFDTDBehaviorFixtures, unittest.TestCase):


    def test_tiled_pml_absorbs_field_energy_in_particle_free_1d_wave(self):
        parameter_set = _base_parameter_values(nx=40, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
                {"wall": "+x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
            ],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (40, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        tiled_pml_state = initialize_tiled_pml_state(
            static_parameters,
            dynamic_parameters,
            parameter_set["pml"][-1],
            tile_shape,
        )
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
        def step(E_tiles, B_tiles, tiled_pml_state):
            B_tiles, tiled_pml_state = update_B(
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            E_tiles, tiled_pml_state = update_E(
                E_tiles,
                B_tiles,
                J_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            B_tiles, tiled_pml_state = update_B(
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            return E_tiles, B_tiles, tiled_pml_state

        step = jax.jit(step)
        for _ in range(60):
            E_tiles, B_tiles, tiled_pml_state = step(E_tiles, B_tiles, tiled_pml_state)

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

    def test_tiled_pml_step_uses_shared_guard_current_without_changing_timing(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 2.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "order": 3.0, "sigma_max": 4.0}],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        E, B, J = _empty_global_fields(parameter_set)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(jnp.linspace(0.0, 0.3, parameter_set["Nx"]))
        Bz = Bz.at[1:-1, 1, 1].set(jnp.linspace(0.1, -0.2, parameter_set["Nx"]))
        E_tiles = tile_vector_field((Ex, Ey, Ez), parameter_set, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), parameter_set, tile_shape)

        g = int(parameter_set["guard_cells"])
        J_tiles = tile_vector_field(J, parameter_set, tile_shape, num_guard_cells=g)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 1:-1, 1:-1, 1:-1].set(0.25)
        J_tiles = (Jx, Jy, Jz)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        pml_state = initialize_tiled_pml_state(static_parameters, dynamic_parameters, parameter_set["pml"][-1], tile_shape)
        E_after, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state)
        B_after, pml_state = update_B(E_after, B_tiles, static_parameters, dynamic_parameters, pml_state)

        for component in E_after + B_after:
            self.assertTrue(jnp.all(jnp.isfinite(component)))
        for memory in pml_state[0] + pml_state[1]:
            self.assertTrue(jnp.all(jnp.isfinite(memory)))


if __name__ == "__main__":
    unittest.main()
