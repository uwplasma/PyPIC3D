"""One-tile API contracts."""


from tests.support.initialization_fixtures import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    InitializationFunctionsFixtures,
    TiledParticles,
    _available_cpu_threads,
    _encode_field_bc,
    _encode_particle_bc,
    _resolve_particle_batch_size,
    build_static_parameters,
    default_parameters,
    initialize_simulation,
    jnp,
    kernel_parameters,
    np,
    os,
    patch,
    physical_fields,
    setup_write_dir,
    tempfile,
    time_loop_electrostatic,
    time_loop_static_metric,
    toml,
    unittest,
    update_parameters_from_toml,
    validate_field_solver,
)


class TestInitializationFunctions(InitializationFunctionsFixtures, unittest.TestCase):
    def test_legacy_gpu_setting_is_accepted_without_changing_runtime_parameters(self):
        template, _ = kernel_parameters()
        for section in ("simulation_parameters", "static_parameters"):
            results = []
            for value in (None, False, True):
                with self.subTest(section=section, GPUs=value):
                    plotting, static, dynamic = default_parameters()
                    static.update(template._asdict())
                    config = {section: {} if value is None else {"GPUs": value}}
                    static, dynamic, plotting = update_parameters_from_toml(
                        toml.loads(toml.dumps(config)), static, dynamic, plotting,
                    )
                    self.assertEqual(static["GPUs"], False if value is None else value)
                    runtime = build_static_parameters(static)
                    self.assertNotIn("GPUs", runtime._fields)
                    results.append((runtime, dynamic, plotting))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])

    def test_static_metric_rejects_conducting_axes_no_wider_than_the_halo(self):
        from PyPIC3D.initialization import _validate_static_metric_conducting_widths
        from tests.kernel_fixtures import kernel_parameters

        def static(solver, boundary_conditions):
            return kernel_parameters(Nx=8, Ny=8, Nz=1, tile_shape=(8, 8, 1), guard_cells=3,
                                     solver=solver, boundary_conditions=boundary_conditions)[0]

        conducting_x = static("static_metric", (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))
        for cells in (1, 3):
            with self.subTest(cells=cells):
                with self.assertRaisesRegex(ValueError, "at least guard_cells"):
                    _validate_static_metric_conducting_widths(conducting_x, (cells, 8, 1))
        _validate_static_metric_conducting_widths(conducting_x, (4, 8, 1))
        # a one-cell periodic axis and the flat Yee solver are not affected
        _validate_static_metric_conducting_widths(static("static_metric", (BC_PERIODIC,) * 3), (8, 8, 1))
        _validate_static_metric_conducting_widths(
            static("electrodynamic_yee", (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)), (2, 8, 1))

    def test_setup_write_dir(self):
        # Should not raise
        setup_write_dir(self.simulation_parameters, self.plotting_parameters)

    def test_default_parameters(self):
        plotting, sim, dynamic = default_parameters()
        self.assertIn('Nx', dynamic)
        self.assertIn('particle_pusher', sim)
        self.assertEqual(sim['particle_pusher'], 'boris')
        self.assertEqual(sim["solver"], "electrodynamic_yee")
        self.assertNotIn("electrostatic", sim)
        self.assertNotIn("fast_backend", sim)
        self.assertEqual(sim["particle_x_bc"], "periodic")
        self.assertEqual(sim["particle_y_bc"], "periodic")
        self.assertEqual(sim["particle_z_bc"], "periodic")
        self.assertIsNone(sim["guard_cells"])  # resolved by build_static_parameters
        self.assertIsNone(sim["particle_batch_size"])
        self.assertFalse(plotting["plotchargedensity"])
        self.assertIn('eps', dynamic)

    def test_automatic_particle_batch_size_uses_all_active_particles_on_one_tile(self):
        active = np.zeros((1, 1, 1, 2, 8), dtype=bool)
        active[0, 0, 0, 0, :5] = True
        active[0, 0, 0, 1, :3] = True
        particles = self._particles_with_active(active)

        batch_size, reason = _resolve_particle_batch_size(particles, None)

        self.assertEqual(batch_size, 8)
        self.assertIn("one tile", reason)


    def test_available_cpu_threads_falls_back_to_os_cpu_count(self):
        with patch("PyPIC3D.initialization.os.sched_getaffinity", side_effect=OSError):
            with patch("PyPIC3D.initialization.os.cpu_count", return_value=12):
                self.assertEqual(_available_cpu_threads(), 12)


    def test_automatic_particle_batch_size_is_one_for_zero_particles(self):
        particles = self._particles_with_active(np.zeros((1, 1, 1, 1, 4), dtype=bool))

        batch_size, _ = _resolve_particle_batch_size(particles, None)

        self.assertEqual(batch_size, 1)

    def test_explicit_particle_batch_size_must_be_a_positive_integer(self):
        particles = self._particles_with_active(np.ones((1, 1, 1, 2, 7), dtype=bool))

        batch_size, _ = _resolve_particle_batch_size(particles, 7)
        clamped_batch_size, _ = _resolve_particle_batch_size(particles, 512)
        self.assertEqual(batch_size, 7)
        self.assertEqual(clamped_batch_size, 14)

        for invalid_batch_size in (0, -1, 1.5, True):
            with self.subTest(particle_batch_size=invalid_batch_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_static_parameters(dict(kernel_parameters()[0]._asdict(), particle_batch_size=invalid_batch_size))

    def test_encode_field_bc_accepts_constant_boundary(self):
        self.assertEqual(_encode_field_bc("constant"), BC_CONSTANT)

    def test_field_and_particle_boundaries_use_one_code_map(self):
        self.assertEqual(BC_PERIODIC, 0)
        self.assertEqual(BC_CONDUCTING, 1)
        self.assertEqual(BC_ABSORBING, 2)
        self.assertEqual(BC_CONSTANT, 3)

        self.assertEqual(_encode_field_bc("periodic"), BC_PERIODIC)
        self.assertEqual(_encode_field_bc("conducting"), BC_CONDUCTING)
        self.assertEqual(_encode_field_bc("constant"), BC_CONSTANT)
        self.assertEqual(_encode_particle_bc("periodic"), BC_PERIODIC)
        self.assertEqual(_encode_particle_bc("reflecting"), BC_CONDUCTING)
        self.assertEqual(_encode_particle_bc("absorbing"), BC_ABSORBING)

    def test_initialize_static_metric_loads_previous_fields_from_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_dtheta_path = os.path.join(tmpdir, "current_dtheta.npy")
            previous_dtheta_path = os.path.join(tmpdir, "previous_dtheta.npy")
            current_bphi_path = os.path.join(tmpdir, "current_bphi.npy")
            previous_bphi_path = os.path.join(tmpdir, "previous_bphi.npy")
            np.save(current_dtheta_path, np.full((4, 4, 1), 2.0))
            np.save(previous_dtheta_path, np.full((4, 4, 1), 3.0))
            np.save(current_bphi_path, np.full((4, 4, 1), 5.0))
            np.save(previous_bphi_path, np.full((4, 4, 1), 7.0))
            config = {
                "simulation_parameters": {
                    "name": "static previous field load test",
                    "output_dir": tmpdir,
                    "solver": "static_metric",
                    "metric": "flat_spherical",
                    "particle_pusher": "hybrid_boris_geodesic",
                    "current_calculation": "GR_direct_deposition",
                    "Nx": 4,
                    "Ny": 4,
                    "Nz": 1,
                    "x_min": 1.0,
                    "x_max": 2.0,
                    "y_min": 0.1,
                    "y_max": 2 * np.pi + 0.1,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-2,
                    "particle_tile_nx": 4,
                    "particle_tile_ny": 4,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                    "x_bc": "constant",
                    "y_bc": "periodic",
                    "z_bc": "periodic",
                    "C": 1.0,
                    "eps": 1.0,
                    "mu": 1.0,
                },
                "plotting": {"plotting": False},
                "field1": {"name": "Dtheta", "type": 1, "path": current_dtheta_path},
                "field2": {"name": "Bphi", "type": 5, "path": current_bphi_path},
                "previous_field1": {"name": "Dtheta previous", "type": 1, "path": previous_dtheta_path},
                "previous_field2": {"name": "Bphi previous", "type": 5, "path": previous_bphi_path},
            }

            loop, particles, fields, static_parameters, dynamic_parameters, *_rest = initialize_simulation(config)

            self.assertIs(loop, time_loop_static_metric)
            g = int(static_parameters.guard_cells)
            D, B, _J, _rho, _phi, _external_fields, _metric, previous_fields, _overflow = physical_fields(fields)
            D_previous, B_previous = previous_fields
            interior = (0, 0, 0, slice(g, -g), slice(g, -g), slice(g, -g))
            self.assertTrue(jnp.allclose(D[1][interior], 2.0))
            self.assertTrue(jnp.allclose(B[2][interior], 5.0))
            self.assertTrue(jnp.allclose(D_previous[1][interior], 3.0))
            self.assertTrue(jnp.allclose(B_previous[2][interior], 7.0))


    def test_initialize_simulation_rejects_one_sided_grid_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "simulation_parameters": {
                    "name": "one sided bounds test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_min": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
                "plotting": {"plotting": False},
            }

            with self.assertRaisesRegex(ValueError, "Both x_min and x_max"):
                initialize_simulation(config)


    def test_initialize_simulation_rejects_unknown_solver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "simulation_parameters": {
                    "name": "unknown solver test",
                    "output_dir": tmpdir,
                    "solver": "old_solver",
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
            }

            with self.assertRaisesRegex(ValueError, "Unsupported solver"):
                initialize_simulation(config)

    def test_validate_field_solver_rejects_spectral(self):
        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("spectral")

    def test_validate_field_solver_accepts_only_public_runtime_modes(self):
        validate_field_solver("electrodynamic_yee")
        validate_field_solver("electrostatic")

        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("fdtd")
        with self.assertRaisesRegex(ValueError, "Unsupported solver"):
            validate_field_solver("tiled_yee")


if __name__ == "__main__":
    unittest.main()
