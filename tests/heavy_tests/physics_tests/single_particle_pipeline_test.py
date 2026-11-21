from tests.support.single_particle_pipeline_runtime_fixtures import (
    Esirkepov_current,
    J_from_rhov,
    _assemble_scalar,
    _assemble_vector,
    _assert_vector_close,
    _empty_electrodynamic_fields,
    _expected_Bz_after_split_update,
    _gather_electric_field,
    _grid_current_work,
    _manual_direct_current_tiles,
    _manual_esirkepov_current_tiles_1d,
    _manual_rho_tiles,
    _one_particle,
    _particle_state,
    _periodic_test_electric_field,
    _runtime_parameters,
    compute_rho,
    empty_tiled_scalar,
    empty_tiled_vector,
    filter_electric_field_for_particles,
    jnp,
    kernel_parameters,
    refresh_tiled_particle_tiles,
    time_loop_electrodynamic,
    unittest,
    update_tiled_particle_positions,
)


class TestSingleParticleStencils(unittest.TestCase):


    def test_direct_current_scatter_is_adjoint_to_production_yee_gather(self):
        velocity = jnp.asarray((0.71, -0.43, 0.29))
        charge = -1.3
        weight = 0.6

        configurations = {
            "one_tile_3d": {
                "grid_shape": (8, 8, 8),
                "tile_shape": (8, 8, 8),
                "wind": (8.0, 8.0, 8.0),
                "positions": {
                    "lower_cell_half": (-3.75, -2.75, -1.75),
                    "upper_periodic_seam": (3.99, 3.91, 3.83),
                },
            },
            "one_tile_reduced": {
                "grid_shape": (8, 1, 1),
                "tile_shape": (8, 1, 1),
                "wind": (8.0, 1.0, 1.0),
                "positions": {
                    "lower_tile_interface": (-0.01, 0.0, 0.0),
                    "upper_periodic_seam": (3.99, 0.0, 0.0),
                },
            },
        }

        parameter_cases = (
            ("one_tile_3d", 1, "none"),
            ("one_tile_3d", 2, "digital"),
            ("one_tile_3d", 1, "bilinear"),
            ("one_tile_reduced", 2, "none"),
            ("one_tile_reduced", 1, "digital"),
            ("one_tile_reduced", 2, "bilinear"),
        )

        for configuration, shape_factor, current_filter in parameter_cases:
            values = configurations[configuration]
            Nx, Ny, Nz = values["grid_shape"]
            x_wind, y_wind, z_wind = values["wind"]
            static_parameters, dynamic_parameters = kernel_parameters(
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                x_wind=x_wind,
                y_wind=y_wind,
                z_wind=z_wind,
                tile_shape=values["tile_shape"],
                guard_cells=2,
                shape_factor=shape_factor,
                current_deposition="direct",
                current_filter=current_filter,
                relativistic=False,
                alpha=0.6,
                dt=0.1,
            )
            E = _periodic_test_electric_field(static_parameters, dynamic_parameters)
            coupling_E = filter_electric_field_for_particles(
                E,
                static_parameters,
                dynamic_parameters,
            )

            for phase, position in values["positions"].items():
                with self.subTest(
                    configuration=configuration,
                    shape_factor=shape_factor,
                    current_filter=current_filter,
                    phase=phase,
                ):
                    particles, species_config = _one_particle(
                        static_parameters,
                        dynamic_parameters,
                        position,
                        velocity,
                        charge=charge,
                        weight=weight,
                    )
                    tile, _, _ = _particle_state(particles)
                    J = J_from_rhov(
                        particles,
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )

                    grid_work = _grid_current_work(
                        E,
                        J,
                        static_parameters,
                        dynamic_parameters,
                    )
                    particle_field = _gather_electric_field(
                        coupling_E,
                        tile,
                        position,
                        static_parameters,
                        dynamic_parameters,
                    )
                    particle_work = charge * weight * jnp.dot(velocity, particle_field)

                    work_scale = max(abs(float(grid_work)), abs(float(particle_work)), 1.0e-30)
                    relative_residual = abs(float(grid_work - particle_work)) / work_scale

                    self.assertLessEqual(relative_residual, 1.0e-12)

    def test_esirkepov_scatter_is_adjoint_to_galerkin_yee_gather(self):
        """The Galerkin gather is the transpose of the Esirkepov scatter.

        Esirkepov builds ``J_x`` from a cumulative sum along ``x``, so its
        effective shape is one order lower along that axis.  Gathering ``E_x``
        with the matching reduced order therefore makes the grid work
        ``sum(E . J) dV`` equal the particle work ``q w v . E`` exactly, which
        is what keeps the field energy consistent.  The deposited current is
        the time average over the step, so the comparison is made at the
        midpoint of the move; the equality is exact for ``shape_factor=1``
        because the reduced shape is then piecewise constant, and second-order
        accurate in ``dt`` for the quadratic ``shape_factor=2`` shapes.
        """

        velocity = jnp.asarray((0.71, -0.43, 0.29))
        charge = -1.3
        weight = 0.6
        dt = 0.01

        configurations = {
            "one_tile_3d": {
                "grid_shape": (8, 8, 8),
                "wind": (8.0, 8.0, 8.0),
                "moving_axes": jnp.asarray((1.0, 1.0, 1.0)),
                "positions": {
                    "lower_cell_half": (-3.75, -0.79, 0.49),
                    "upper_periodic_seam": (3.99, 3.91, 3.83),
                },
            },
            "one_tile_reduced": {
                "grid_shape": (8, 1, 1),
                "wind": (8.0, 1.0, 1.0),
                "moving_axes": jnp.asarray((1.0, 0.0, 0.0)),
                "positions": {
                    "lower_tile_interface": (-3.30, 0.0, 0.0),
                    "upper_periodic_seam": (3.99, 0.0, 0.0),
                },
            },
        }

        for configuration, values in configurations.items():
            Nx, Ny, Nz = values["grid_shape"]
            x_wind, y_wind, z_wind = values["wind"]
            # the Esirkepov gather is selected by the deposition scheme, so the
            # full-order reference is the same run with direct deposition
            galerkin_parameters, dynamic_parameters = kernel_parameters(
                Nx=Nx, Ny=Ny, Nz=Nz,
                x_wind=x_wind, y_wind=y_wind, z_wind=z_wind,
                tile_shape=values["grid_shape"],
                guard_cells=2,
                shape_factor=1,
                current_deposition="esirkepov",
                current_filter="none",
                relativistic=False,
                dt=dt,
            )
            full_order_parameters = galerkin_parameters._replace(current_deposition="direct")
            E = _periodic_test_electric_field(galerkin_parameters, dynamic_parameters)

            for phase, position in values["positions"].items():
                with self.subTest(configuration=configuration, phase=phase):
                    particles, species_config = _one_particle(
                        galerkin_parameters,
                        dynamic_parameters,
                        position,
                        velocity,
                        charge=charge,
                        weight=weight,
                    )
                    displacement = dt * velocity * values["moving_axes"]
                    moved = particles._replace(x=particles.x + displacement)
                    J = Esirkepov_current(
                        particles,
                        moved,
                        species_config,
                        empty_tiled_vector(galerkin_parameters, dynamic_parameters),
                        galerkin_parameters,
                        dynamic_parameters,
                        coordinate_velocity=jnp.broadcast_to(velocity, particles.x.shape),
                    )

                    grid_work = _grid_current_work(
                        E,
                        J,
                        galerkin_parameters,
                        dynamic_parameters,
                    )

                    tile, start, _ = _particle_state(particles)
                    midpoint = start + 0.5 * displacement

                    def residual(parameters):
                        particle_field = _gather_electric_field(
                            E,
                            tile,
                            midpoint,
                            parameters,
                            dynamic_parameters,
                        )
                        particle_work = charge * weight * jnp.dot(velocity, particle_field)
                        work_scale = max(
                            abs(float(grid_work)), abs(float(particle_work)), 1.0e-30
                        )
                        return abs(float(grid_work - particle_work)) / work_scale

                    self.assertLessEqual(residual(galerkin_parameters), 1.0e-10)
                    # the uniform-order gather is not the adjoint, so it must
                    # miss by far more than the tolerance above
                    self.assertGreater(residual(full_order_parameters), 1.0e-3)


class TestSingleParticleElectrodynamicPipeline(unittest.TestCase):
    def test_direct_current_source_propagates_into_E_and_B(self):
        static_parameters, dynamic_parameters = _runtime_parameters(
            shape_factor=2,
            current_deposition="direct",
        )
        particles, species_config = _one_particle(
            static_parameters,
            dynamic_parameters,
            (-1.32, 0.0, 0.0),
            (0.0, 0.2, 0.0),
        )
        fields = _empty_electrodynamic_fields(static_parameters, dynamic_parameters)
        E_before_global = _assemble_vector(fields[0], static_parameters)

        particles_after, fields_after = time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        E_after, B_after, J_after, *_rest = fields_after

        centered_particles = update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt / 2)
        centered_particles, overflow = refresh_tiled_particle_tiles(centered_particles, static_parameters, dynamic_parameters)
        self.assertFalse(bool(overflow))
        expected_J = _manual_direct_current_tiles(centered_particles, species_config, static_parameters, dynamic_parameters)
        _assert_vector_close(self, J_after, expected_J)

        E_global = _assemble_vector(E_after, static_parameters)
        J_global = _assemble_vector(expected_J, static_parameters)
        for E_component, J_component in zip(E_global, J_global):
            self.assertTrue(
                jnp.allclose(
                    E_component[1:-1, 1:-1, 1:-1],
                    -dynamic_parameters.dt * J_component[1:-1, 1:-1, 1:-1] / dynamic_parameters.eps,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

        B_global = _assemble_vector(B_after, static_parameters)
        expected_Bz = _expected_Bz_after_split_update(
            E_before_global[1],
            E_global[1],
            dynamic_parameters,
        )
        self.assertTrue(jnp.allclose(B_global[0][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(B_global[1][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(
            jnp.allclose(
                B_global[2][1:-1, 1:-1, 1:-1],
                expected_Bz[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        self.assertFalse(bool(fields_after[-1]))
        self.assertEqual(int(jnp.sum(particles_after.active)), 1)

    def test_esirkepov_current_source_propagates_into_E_and_B(self):
        static_parameters, dynamic_parameters = _runtime_parameters(
            shape_factor=1,
            current_deposition="esirkepov",
        )
        initial_x = (-1.32, 0.0, 0.0)
        initial_u = (0.08, 0.2, 0.0)
        particles, species_config = _one_particle(static_parameters, dynamic_parameters, initial_x, initial_u)
        fields = _empty_electrodynamic_fields(static_parameters, dynamic_parameters)
        E_before_global = _assemble_vector(fields[0], static_parameters)

        particles_after, fields_after = time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        E_after, B_after, J_after, *_rest = fields_after

        expected_J = _manual_esirkepov_current_tiles_1d(particles, species_config, static_parameters, dynamic_parameters)
        _assert_vector_close(self, J_after, expected_J)

        active_x = particles_after.x[..., 0][particles_after.active]
        self.assertTrue(
            jnp.allclose(
                active_x,
                jnp.asarray([initial_x[0] + initial_u[0] * float(dynamic_parameters.dt)]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

        E_global = _assemble_vector(E_after, static_parameters)
        J_global = _assemble_vector(expected_J, static_parameters)
        for E_component, J_component in zip(E_global, J_global):
            self.assertTrue(
                jnp.allclose(
                    E_component[1:-1, 1:-1, 1:-1],
                    -dynamic_parameters.dt * J_component[1:-1, 1:-1, 1:-1] / dynamic_parameters.eps,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

        B_global = _assemble_vector(B_after, static_parameters)
        expected_Bz = _expected_Bz_after_split_update(
            E_before_global[1],
            E_global[1],
            dynamic_parameters,
        )
        self.assertTrue(jnp.allclose(B_global[0][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(B_global[1][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(
            jnp.allclose(
                B_global[2][1:-1, 1:-1, 1:-1],
                expected_Bz[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        self.assertFalse(bool(fields_after[-1]))


if __name__ == "__main__":
    unittest.main()
