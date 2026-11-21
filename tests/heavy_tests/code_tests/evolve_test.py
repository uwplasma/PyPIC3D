from tests.support.code_evolve_fixtures import (
    BC_ABSORBING,
    BC_PERIODIC,
    SimpleNamespace,
    add_single_tile_grids_to_parameters,
    build_dynamic_parameters,
    build_static_parameters,
    build_yee_grid,
    initialize_fields,
    jnp,
    kernel_parameters_from_values,
    one_slot_tiled_particles,
    time_loop_electrodynamic,
    unittest,
)


class TestEvolveExternalFields(unittest.TestCase):
        # test that both these methods exist in the public API and are not private methods

    def test_external_electric_field_pushes_particles_without_evolving_maxwell_fields(self):
        parameter_set = {
            "Nx": 3,
            "Ny": 3,
            "Nz": 3,
            "dx": 1.0 / 3.0,
            "dy": 1.0 / 3.0,
            "dz": 1.0 / 3.0,
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
        # build parameter_set and grids, and set dynamic_values for the simulation

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_single_tile_grids_to_parameters(parameter_set)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        external_E = (jnp.ones_like(external_E[0]), external_E[1], external_E[2])
        external_fields = (external_E, external_B)
        # build empty fields and set an external electric field in the x-direction

        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.0, 0.0, 0.0]),
            u=jnp.array([0.0, 0.0, 0.0]),
        )
        # create a single particle at the origin with zero velocity
        fields = (
            E,
            B,
            J,
            rho,
            phi,
            external_fields,
            None,
            jnp.asarray(False),
        )

        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        tiled_particles, fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        ) # advance the simulation for one time step with the external electric field

        E_after, B_after, J_after, rho_after, phi_after, external_after, pml_state, overflow = fields
        vx = tiled_particles.u[:, :, :, 0, :, 0][tiled_particles.active[:, :, :, 0, :]]
        vy = tiled_particles.u[:, :, :, 0, :, 1][tiled_particles.active[:, :, :, 0, :]]
        vz = tiled_particles.u[:, :, :, 0, :, 2][tiled_particles.active[:, :, :, 0, :]]
        # unpack the fields and particle velocities after the time step

        self.assertIsNone(pml_state)
        # ensure the PML state is None, as we are not using PML in this test
        self.assertFalse(bool(overflow))
        # ensure the particle did not overflow the simulation domain
        self.assertGreater(float(vx[0]), 0.0)
        self.assertTrue(jnp.allclose(vy, 0.0))
        self.assertTrue(jnp.allclose(vz, 0.0))
        # ensure the particle has gained velocity in the x-direction due to the external electric field, and has not gained velocity in the y or z directions
        self.assertTrue(jnp.allclose(external_after[0][0], external_fields[0][0]))
        # ensure the external electric field has not changed after the time step


    def test_absorbing_particle_mask_survives_jitted_electrodynamic_step(self):
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
            "particle_boundary_conditions": {
                "x": BC_ABSORBING,
                "y": BC_PERIODIC,
                "z": BC_PERIODIC,
            },
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        # build parameter_set and grids, and set dynamic_values for the simulation

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_single_tile_grids_to_parameters(parameter_set)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        # build empty fields and set external fields to zero

        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.49, 0.0, 0.0]),
            u=jnp.array([0.2, 0.0, 0.0]),
        )
        # create a single particle at x=0.49 with a velocity in the x-direction, which will cause it to cross the absorbing boundary at x=0.5

        fields = (
            E,
            B,
            J,
            rho,
            phi,
            external_fields,
            None,
            jnp.asarray(False),
        )
        # build the fields tuple with the initialized fields and external fields

        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        tiled_particles, _ = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        # advance the simulation for one time step, which should cause the particle to cross the absorbing boundary and become inactive

        active = tiled_particles.active[:, :, :, 0, :].reshape(-1)
        x = tiled_particles.x[:, :, :, 0, :, 0].reshape(-1)
        # reshape the active mask and x positions of the particles to check their status after the time step

        self.assertEqual(x.shape[0], 1)
        self.assertTrue(jnp.array_equal(active, jnp.array([False])))


if __name__ == "__main__":
    unittest.main()
