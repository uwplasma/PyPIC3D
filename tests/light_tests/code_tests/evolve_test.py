from tests.support.code_evolve_fixtures import (
    SimpleNamespace,
    add_single_tile_grids_to_parameters,
    build_dynamic_parameters,
    build_static_parameters,
    build_yee_grid,
    initialize_fields,
    jax,
    jnp,
    kernel_parameters_from_values,
    one_slot_tiled_particles,
    time_loop_electrodynamic,
    time_loop_electrostatic,
    unittest,
)


class TestEvolveExternalFields(unittest.TestCase):
    def test_public_loop_names_own_tiled_contracts(self):
        self.assertEqual(time_loop_electrodynamic.__module__, "PyPIC3D.solvers.yee.time_loop")
        self.assertEqual(time_loop_electrostatic.__module__, "PyPIC3D.solvers.electrostatic.time_loop")
        # test that both these methods exist in the public API and are not private methods

        # ensure the external electric field has not changed after the time step

    def test_electrodynamic_step_accepts_split_kernel_parameters(self):
        parameter_set = {
            "Nx": 3,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0 / 3.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "current_deposition": "direct",
            "current_filter": "none",
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_single_tile_grids_to_parameters(parameter_set)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        fields = (E, B, J, rho, phi, external_fields, None, jnp.asarray(False))
        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.0, 0.0, 0.0]),
            u=jnp.array([0.05, 0.0, 0.0]),
        )
        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        step = jax.jit(lambda particles, fields, dynamic: time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic,
        ))
        tiled_particles, fields = step(tiled_particles, fields, dynamic_parameters)

        self.assertFalse(bool(fields[-1]))
        self.assertTrue(jnp.all(tiled_particles.active))
        self.assertGreater(float(tiled_particles.x[0, 0, 0, 0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
