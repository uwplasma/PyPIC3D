"""Single-device numerical tests."""

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
    contextlib,
    default_parameters,
    initialize_simulation,
    io,
    jax,
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


    def test_automatic_particle_batch_size_uses_cpu_threads_per_local_device(self):
        active = np.zeros((2, 2, 1, 1, 1000), dtype=bool)
        active[..., :900] = True
        particles = self._particles_with_active(active)

        wider, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=16,
            local_device_count=4,
        )
        narrower, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=8,
            local_device_count=4,
        )

        self.assertEqual(wider, 512)
        self.assertEqual(narrower, 256)


    def test_available_cpu_threads_falls_back_to_os_cpu_count(self):
        with patch("PyPIC3D.initialization.os.sched_getaffinity", side_effect=OSError):
            with patch("PyPIC3D.initialization.os.cpu_count", return_value=12):
                self.assertEqual(_available_cpu_threads(), 12)


    def test_automatic_particle_batch_size_uses_accelerator_fallback(self):
        active = np.zeros((2, 1, 1, 1, 2048), dtype=bool)
        active[..., :1500] = True
        particles = self._particles_with_active(active)

        batch_size, reason = _resolve_particle_batch_size(particles, None, platform="gpu")

        self.assertEqual(batch_size, 1024)
        self.assertIn("gpu", reason)


    def test_automatic_particle_batch_size_uses_largest_active_tile(self):
        active = np.zeros((2, 1, 1, 1, 32), dtype=bool)
        active[0, 0, 0, 0, :7] = True
        active[1, 0, 0, 0, :13] = True
        particles = self._particles_with_active(active)

        batch_size, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=64,
            local_device_count=2,
        )

        self.assertEqual(batch_size, 13)


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


    def test_filtered_leapfrog_seed_matches_force_and_zero_dark_coupling(self):
        # Use Ey on center-grid nodes: the independent gather is the node value.
        n, dt, alpha, charge, mass, c = 8, .01, .4, 2., 4., 3.
        x = -0.5 + np.arange(n) / n
        evolved = .6 * np.cos(2 * np.pi * np.arange(n) / n)
        external = .3 * np.sin(4 * np.pi * np.arange(n) / n) + .1
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            def save(name, data):
                path = os.path.join(directory, name + ".npy")
                np.save(path, data)
                return path

            x_path = save("x", x)
            evolved_path = save("evolved", evolved.reshape(n, 1, 1))
            external_path = save("external", external.reshape(n, 1, 1))
            for pusher in ("boris", "higuera_cary"):
                for filter_name in ("none", "digital", "bilinear"):
                    if filter_name == "bilinear":
                        filtered = (np.roll(evolved, 1) + 2 * evolved + np.roll(evolved, -1)) / 4
                    elif filter_name == "digital":
                        filtered = alpha * evolved + (1 - alpha) / 6 * (
                            np.roll(evolved, 1) + np.roll(evolved, -1) + 4 * evolved
                        )
                    else:
                        filtered = evolved
                    momentum = -(charge / mass) * (filtered + external) * dt / 2
                    expected = momentum if pusher == "boris" else momentum / np.sqrt(1 + (momentum / c)**2)
                    states = []
                    for solver in ("electrodynamic_yee", "dark_matter_yee"):
                        with self.subTest(pusher=pusher, filter=filter_name, solver=solver):
                            config = {
                                "simulation_parameters": dict(
                                    name="filtered leapfrog seed", output_dir=directory,
                                    solver=solver, Nx=n, Ny=1, Nz=1,
                                    x_wind=1., y_wind=1., z_wind=1., dt=dt, Nt=1,
                                    particle_tile_nx=n, particle_tile_ny=1, particle_tile_nz=1,
                                    shape_factor=1, particle_pusher=pusher, relativistic=False,
                                    current_calculation="j_from_rhov", filter_j=filter_name,
                                    alpha=alpha, C=c, eps=1., mu=1/c**2, sin_chi=0., dark_mu=0.,
                                ),
                                "particle1": dict(
                                    name="p", N_particles=n, charge=charge, mass=mass, temperature=0.,
                                    initial_x=x_path, initial_y=0., initial_z=0.,
                                    initial_vx=0., initial_vy=0., initial_vz=0.,
                                ),
                                "field1": dict(name="evolved Ey", type=1, path=evolved_path, evolve=True),
                                "field2": dict(name="external Ey", type=1, path=external_path, evolve=False),
                            }
                            _, particles, *_ = initialize_simulation(config)
                            positions = np.asarray(particles.x)[np.asarray(particles.active)]
                            velocities = np.asarray(particles.u)[np.asarray(particles.active)]
                            order = np.argsort(positions[:, 0])
                            np.testing.assert_allclose(positions[order, 0], x, atol=1e-15)
                            np.testing.assert_array_equal(positions[:, 1:], 0.)
                            np.testing.assert_allclose(velocities[order, 1], expected, rtol=1e-11, atol=1e-14)
                            np.testing.assert_array_equal(velocities[:, (0, 2)], 0.)
                            states.append(velocities[order])
                    np.testing.assert_array_equal(states[0], states[1])


    def test_initialize_simulation_offsets_particle_velocity_to_the_half_step(self):
        """
        The time loops advance ``u^{n-1/2}`` to ``u^{n+1/2}`` with the force at
        ``x^n``, so the run has to start from ``u(-dt/2)``.  Handing the
        configured ``u(0)`` straight in leaves an O(dt) error in the initial
        state and drops the whole simulation to first order.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            Nx, Ny, Nz = 4, 1, 1
            dt = 1.0e-3
            charge, mass, Ex = -2.0, 4.0, 0.75

            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            field_path = os.path.join(tmpdir, "ex.npy")
            np.save(x_path, np.array([-0.375, -0.125, 0.125, 0.375]))
            np.save(zeros_path, np.zeros(4))
            np.save(field_path, np.full((Nx, Ny, Nz), Ex))

            config = {
                "simulation_parameters": {
                    "name": "leapfrog velocity seed",
                    "output_dir": tmpdir,
                    "Nx": Nx,
                    "Ny": Ny,
                    "Nz": Nz,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": dt,
                    "particle_tile_nx": Nx,
                    "particle_tile_ny": Ny,
                    "particle_tile_nz": Nz,
                    "filter_j": "none",
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": charge,
                    "mass": mass,
                    "temperature": 0.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
                "field1": {
                    "name": "uniform Ex",
                    "type": 0,
                    "path": field_path,
                    "evolve": False,
                },
            }

            _loop, particles, *_rest = initialize_simulation(config)

            u = np.asarray(jax.device_get(particles.u))[particles.active]
            expected_vx = -(charge / mass) * Ex * 0.5 * dt
            # a particle at rest feels only the electric kick over -dt/2
            np.testing.assert_allclose(
                u[:, 0], expected_vx, rtol=1.0e-6,
                err_msg="initialize_simulation did not offset u onto the leapfrog half step",
            )
            np.testing.assert_allclose(u[:, 1], 0.0, atol=1.0e-15)
            np.testing.assert_allclose(u[:, 2], 0.0, atol=1.0e-15)


    def test_initialize_simulation_computes_courant_dt_before_runtime_parameters_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            np.save(x_path, np.array([-0.375, -0.125, 0.125, 0.375]))
            np.save(zeros_path, np.zeros(4))
            config = {
                "simulation_parameters": {
                    "name": "courant dt tiled runtime test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "particle_tile_nx": 4,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                },
                "plotting": {
                    "plotvelocities": True,
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            config_path = os.path.join(tmpdir, "courant_dt.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            result = initialize_simulation(toml.load(config_path))
            dynamic_parameters = result[4]
            plotting_parameters = result[5]

            self.assertGreater(float(dynamic_parameters.dt), 0.0)
            self.assertEqual(
                tuple(plotting_parameters["field_map"]),
                ("E", "B", "J", "fluid_velocity"),
            )
            for velocity_component in plotting_parameters["field_map"]["fluid_velocity"]:
                self.assertTrue(jnp.allclose(velocity_component, 0.0))


    def test_initialize_simulation_builds_grid_from_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            z_path = os.path.join(tmpdir, "z.npy")
            np.save(x_path, np.array([1.25, 1.75, 2.25, 2.75]))
            np.save(zeros_path, np.zeros(4))
            np.save(z_path, np.full(4, 2.5))
            config = {
                "simulation_parameters": {
                    "name": "shifted grid bounds test",
                    "output_dir": tmpdir,
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_min": 1.0,
                    "x_max": 3.0,
                    "y_min": -0.5,
                    "y_max": 0.5,
                    "z_min": 2.0,
                    "z_max": 3.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                    "particle_tile_nx": 4,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                },
                "plotting": {"plotting": False},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": z_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            result = initialize_simulation(config)
            dynamic_parameters = result[4]

            self.assertAlmostEqual(float(dynamic_parameters.dx), 0.5)
            self.assertAlmostEqual(float(dynamic_parameters.x_wind), 2.0)
            self.assertTrue(jnp.allclose(
                dynamic_parameters.grids.center[0],
                jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            ))
            self.assertTrue(jnp.allclose(
                dynamic_parameters.grids.vertex[0],
                jnp.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.25]),
            ))
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[1][1]), -0.5)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[1][-1]), 0.5)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[2][1]), 2.0)
            self.assertAlmostEqual(float(dynamic_parameters.grids.center[2][-1]), 3.0)


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


    def test_initialize_simulation_encodes_global_particle_boundary_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            np.save(zeros_path, np.zeros(1))
            config = {
                "simulation_parameters": {
                    "name": "global particle bc test",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 1,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                    "particle_x_bc": "reflecting",
                    "particle_y_bc": "absorbing",
                    "particle_z_bc": "periodic",
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 1,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "x_bc": "absorbing",
                    "initial_x": zeros_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            config_path = os.path.join(tmpdir, "global_particle_bc.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            _, particles, _, parameter_set, *_ = initialize_simulation(toml.load(config_path))

            self.assertEqual(
                parameter_set.particle_boundary_conditions,
                (BC_CONDUCTING, BC_ABSORBING, BC_PERIODIC),
            )
            self.assertIsInstance(particles, TiledParticles)


    def test_initialize_simulation_uses_collocated_grid_for_electrostatic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            np.save(zeros_path, np.zeros(1))
            config = {
                "simulation_parameters": {
                    "name": "electrostatic collocated grid test",
                    "output_dir": tmpdir,
                    "solver": "electrostatic",
                    "Nx": 4,
                    "Ny": 2,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 1,
                    "charge": -1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": zeros_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            loop, particles, fields, parameter_set, dynamic_parameters, *_ = initialize_simulation(config)

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            for vertex_axis, center_axis in zip(dynamic_parameters.grids.vertex, dynamic_parameters.grids.center):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))


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
