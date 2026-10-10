"""Single-device numerical tests."""

from tests.support.dark_photon_fixtures import (
    D_FIELD_LOCATIONS,
    DarkPhotonFixtures,
    _dark_timestep,
    build_field_output_map,
    build_static_parameters,
    build_tiled_particles,
    compute_dark_B,
    compute_dark_energy,
    contextlib,
    dark_divergence,
    dark_gradient,
    dark_photon_push_fields,
    default_parameters,
    evolve,
    field_map_for_output,
    filter_electric_field_for_particles,
    initialize_dark_photon_fields,
    initialize_fields,
    initialize_simulation,
    interior,
    io,
    jax,
    jnp,
    kernel_parameters,
    load_dark_fields_from_toml,
    math,
    np,
    os,
    parameters,
    particle_species,
    pmd,
    run_PyPIC3D,
    static_parameters_for_output,
    synchronized_dark_fields,
    tempfile,
    time_loop_dark_photon,
    time_loop_electrodynamic,
    toml,
    unittest,
    update_dark_E,
    update_tiled_ghost_cells,
    update_tiled_vector_ghost_cells,
    wave,
    wave_errors,
)


class TestDarkPhoton(DarkPhotonFixtures, unittest.TestCase):
    def test_transverse_and_longitudinal_free_waves(self):
        for longitudinal in (False, True):
            with self.subTest(longitudinal=longitudinal):
                s, d = parameters(n=128, dt=1/512)
                E, A, phi = wave(s, d, longitudinal)
                initial = initialize_dark_photon_fields(s, d, E, A, phi)
                result = evolve(initial, s, d, 512)
                self.assertLess(max(wave_errors(s, d, result, longitudinal)), 3e-4)
                if longitudinal:
                    for b in compute_dark_B(result[1], s, d):
                        np.testing.assert_allclose(b, 0., atol=1e-13)
                else:
                    np.testing.assert_allclose(result[2], 0., atol=1e-13)


    def test_second_order_spatial_convergence(self):
        for longitudinal in (False, True):
            errors = []
            for n in (32, 64, 128):
                s, d = parameters(n, dt=1/4096)
                state = initialize_dark_photon_fields(s, d, *wave(s, d, longitudinal))
                result = evolve(state, s, d, 4096)
                errors.append(wave_errors(s, d, result, longitudinal))
            orders = np.log2(np.array(errors[:-1])/errors[1:])
            with self.subTest(longitudinal=longitudinal, orders=orders):
                self.assertTrue(np.all((orders > 1.8) & (orders < 2.2)), orders)


    def test_second_order_temporal_convergence_without_spatial_error_floor(self):
        for longitudinal in (False, True):
            errors = []
            for steps in (32, 64, 128):
                s, d = parameters(dt=1/steps)
                state = initialize_dark_photon_fields(s, d, *wave(s, d, longitudinal, discrete=True))
                errors.append(wave_errors(s, d, evolve(state, s, d, steps), longitudinal, discrete=True))
            orders = np.log2(np.array(errors[:-1])/errors[1:])
            with self.subTest(longitudinal=longitudinal, orders=orders):
                self.assertTrue(np.all((orders > 1.8) & (orders < 2.2)), orders)


    def test_uniform_mass_oscillation_and_massless_limit(self):
        for mass in (0., 0.7):
            s, d = parameters(n=1, dt=1/1024, dark_mu=mass)
            E, A, J, phi, _ = initialize_fields(s, d)
            A = (jnp.ones_like(A[0]), A[1], A[2])
            state = initialize_dark_photon_fields(s, d, E, A, phi)
            result = evolve(state, s, d, 1024)
            omega = float(d.C)*mass
            np.testing.assert_allclose(result[0][0][interior(s)], omega*np.sin(omega), atol=2e-7)
            np.testing.assert_allclose(result[1][0][interior(s)], np.cos(omega*(1-float(d.dt)/2)), atol=2e-7)
        s, d = parameters(dark_mu=0., dt=1/128)
        state = initialize_dark_photon_fields(s, d, *wave(s, d, False, discrete=True))
        self.assertLess(max(wave_errors(s, d, evolve(state, s, d, 128), False, discrete=True)), 3e-5)


    def test_yee_identities_and_gauss_constraint(self):
        for g in (1, 2):
            s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, guard_cells=g, dark_mu=.7, dt=.001)
            zero = initialize_fields(s, d)[3]
            noise = jnp.asarray(np.random.default_rng(42).normal(size=zero.shape))
            noise = update_tiled_ghost_cells(noise, s, g, location=("C", "C", "C"))
            E = dark_gradient(noise, s, d)
            E = update_tiled_vector_ghost_cells(E, s, g, locations=D_FIELD_LOCATIONS)
            for component in compute_dark_B(E, s, d):
                np.testing.assert_allclose(component[interior(s)], 0., atol=4e-14)
            phi = -dark_divergence(E, s, d)/s.dark_mu**2
            state = initialize_dark_photon_fields(s, d, E, (noise, noise, noise), phi)
            result = evolve(state, s, d, 12)
            residual = dark_divergence(result[0], s, d)+s.dark_mu**2*result[2]
            np.testing.assert_allclose(residual[interior(s)], 0., atol=2e-13)


    def test_current_source_and_filtered_particle_force_signs(self):
        for mixing in (-.2, .3):
            for filter_name in ("none", "bilinear", "digital"):
                s, d = parameters(sin_chi=mixing, dark_mu=0., eps=2.5, current_filter=filter_name, alpha=.5)
                E, B, J, phi, _ = initialize_fields(s, d)
                current = (jnp.ones_like(J[0]), J[1], J[2])
                result = update_dark_E(E, B, B, current, s, d, d.dt)
                np.testing.assert_allclose(result[0][interior(s)], mixing*float(d.dt)/2.5, atol=1e-15)
                dark = initialize_dark_photon_fields(s, d, *wave(s, d, False))
                dark_E, _, _, dark_B = synchronized_dark_fields(dark, s, d)
                external = (current, current)
                push_E, push_B = dark_photon_push_fields(E, B, dark, external, s, d)
                filtered = filter_electric_field_for_particles(dark_E, s, d)
                self.assert_tree_close(push_E, tuple(ext-mixing*e for ext, e in zip(current, filtered)))
                self.assert_tree_close(push_B, tuple(ext-mixing*b for ext, b in zip(current, dark_B)))


    def test_zero_mixing_time_loop_matches_yee(self):
        steps = 5
        for deposition, filter_name in (("direct", "bilinear"), ("direct", "digital"), ("esirkepov", "none")):
            with self.subTest(deposition=deposition, filter=filter_name):
                s, d = parameters(n=8, sin_chi=0., current_deposition=deposition,
                                  current_filter=filter_name, alpha=.5)
                particles, species = build_tiled_particles([
                    particle_species("p", charge=.01, mass=1., x1=[-.4, .4], u1=[.02, -.03]),
                ], s, d)
                E, B, J, phi, rho = initialize_fields(s, d)
                external = (tuple(jnp.ones_like(e)*.01 for e in E), B)
                dark = initialize_dark_photon_fields(s, d, *wave(s, d, False))
                yee_state = (particles, (E, B, J, rho, phi, external, None, jnp.asarray(False)))
                dark_state = (particles, (E, B, J, rho, phi, external, None, dark, jnp.asarray(False)))

                yee_step = jax.jit(lambda p, f: time_loop_electrodynamic(p, species, f, s, d))
                dark_step = jax.jit(lambda p, f: time_loop_dark_photon(p, species, f, s, d))
                eager = time_loop_dark_photon(dark_state[0], species, dark_state[1], s, d)
                self.assert_tree_close(eager, dark_step(*dark_state))
                for _ in range(steps):
                    yee_state = yee_step(*yee_state)
                    dark_state = dark_step(*dark_state)

                dark_particles, dark_fields = dark_state
                self.assert_tree_close(yee_state, (dark_particles, dark_fields[:7] + (dark_fields[8],)))
                self.assert_tree_close(dark_fields[7], evolve(dark, s, d, steps))


    def test_particle_electric_force_is_scaled_and_signed(self):
        for deposition in ("direct", "esirkepov"):
            s, d = parameters(n=8, sin_chi=.2, dark_mu=0., current_deposition=deposition, relativistic=False)
            particles, species = build_tiled_particles([
                particle_species("p", charge=2., mass=4., x1=[0.], u1=[0.]),
            ], s, d)
            E, B, J, phi, rho = initialize_fields(s, d)
            dark_E = (jnp.ones_like(E[0])*.75, E[1], E[2])
            dark = initialize_dark_photon_fields(s, d, dark_E)
            fields = (E, B, J, rho, phi, (E, B), None, dark, jnp.asarray(False))
            result, _ = time_loop_dark_photon(particles, species, fields, s, d)
            np.testing.assert_allclose(np.asarray(result.u)[np.asarray(result.active), 0],
                                       -(2/4)*.2*.75*float(d.dt), atol=1e-14)


    def test_toml_initialization_seed_output_energy_and_driver(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = self.config(directory)
            config["plotting"] = dict(dump_fields=True, plot_openpmd_fields=True)
            for field_type, amplitude in ((0, .25), (3, .5), (6, .1)):
                path = os.path.join(directory, f"initial_{field_type}.npy")
                np.save(path, np.full((8, 1, 1), amplitude))
                config[f"dark_field{field_type}"] = dict(name="dark", type=field_type, path=path)
            config = toml.loads(toml.dumps(config))
            loop, particles, fields, s, d, plotting, _, species = initialize_simulation(config)
            self.assertIs(loop, time_loop_dark_photon)
            self.assertEqual(len(fields), 9)
            self.assertEqual(s.sin_chi, .2)
            self.assertEqual(s.dark_mu, .7)
            self.assertEqual(static_parameters_for_output(s)["dark_mu"], .7)
            self.assertEqual(build_static_parameters(s._asdict()), s)
            self.assertEqual(toml.loads(toml.dumps(static_parameters_for_output(s)))["sin_chi"], .2)
            E, A, phi = fields[7]
            np.testing.assert_allclose(A[0], .5+.25*float(d.dt)/2, atol=1e-15)
            snapshot = build_field_output_map(fields, particles, species, s, d)
            np.testing.assert_allclose(snapshot["dark_A"][0], .5, atol=1e-15)
            self.assertEqual(set(snapshot), {"E", "B", "J", "dark_E", "dark_A", "dark_phi", "dark_B"})
            np.testing.assert_allclose(field_map_for_output(snapshot, s)["dark_A"][0], .5, atol=1e-15)
            expected_energy = .5*2*(2*math.pi)*(.25**2+1.3**2*.7**2*.5**2+.7**2*.1**2)
            self.assertAlmostEqual(float(compute_dark_energy(fields[7], s, d)), expected_energy, places=12)
            config["plotting"]["dump_fields"] = False
            result = run_PyPIC3D(config)
            self.assertEqual(len(result[5]), 9)
            recorded = np.loadtxt(os.path.join(directory, "data", "total_energy.txt"), delimiter=",")
            self.assertAlmostEqual(float(np.ravel(recorded)[1]), expected_energy, places=10)
            for filename in (os.path.join("initial_fields", "initial_fields.h5"), "fields.h5"):
                series = pmd.Series(os.path.join(directory, "data", filename), pmd.Access.read_only)
                iteration = series.iterations[0]
                for name in ("dark_E", "dark_A", "dark_phi", "dark_B"):
                    self.assertIn(name, iteration.meshes)
                    self.assertEqual(iteration.meshes[name].time_offset, 0.)
                np.testing.assert_allclose(iteration.meshes["dark_A"]["x"].position, [.5, 0., 0.])
                np.testing.assert_allclose(iteration.meshes["dark_B"]["x"].position, [0., .5, .5])
                values = iteration.meshes["dark_A"]["x"].load_chunk()
                series.flush()
                np.testing.assert_allclose(values, .5, atol=1e-15)
                series.close()


    def test_initial_particle_velocity_uses_dark_force(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = self.config(directory, dark_mu=0., relativistic=False)
            path = os.path.join(directory, "electric.npy")
            np.save(path, np.full((8, 1, 1), .75))
            config["dark_field0"] = dict(type=0, path=path)
            config["particle1"] = dict(name="p", N_particles=1, mass=4., charge=2., temperature=0.,
                                       initial_x=0., initial_y=0., initial_z=0.,
                                       initial_vx=0., initial_vy=0., initial_vz=0.)
            _, particles, _, _, d, *_ = initialize_simulation(config)
            np.testing.assert_allclose(np.asarray(particles.u)[np.asarray(particles.active), 0],
                                       (2/4)*.2*.75*float(d.dt)/2, atol=1e-15)


    def test_configuration_errors_and_timestep_bound(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for updates in (dict(sin_chi=1.1), dict(sin_chi=float("nan")), dict(dark_mu=-1.),
                            dict(dark_mu=float("inf")), dict(y_bc="constant"),
                            dict(dt=-1.), dict(dt=float("nan")), dict(dt=100.), dict(C=0.), dict(eps=0.)):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    initialize_simulation(self.config(directory, **updates))
            for absorber in ("supergaussian",):
                config = self.config(directory)
                config[absorber] = [{"axis": "x"}]
                with self.assertRaisesRegex(ValueError, "does not support"):
                    initialize_simulation(config)
            for updates in (dict(sin_chi=2.), dict(dark_mu=-1.),
                            dict(supergaussian_active=True), dict(boundary_conditions=(2, 0, 0))):
                s, _ = parameters()
                with self.assertRaises(ValueError):
                    build_static_parameters({**s._asdict(), **updates})
        _, static, dynamic = default_parameters()
        dynamic.update(Nx=8, Ny=1, Nz=1, dx=.2, dy=1., dz=1., C=1.3, dt=None)
        static.update(dark_mu=20., sin_chi=.1)
        limit = 2/(1.3*math.sqrt(20**2+4/.2**2))
        self.assertAlmostEqual(_dark_timestep(static, dynamic), .99*limit)
        dynamic["dt"] = limit
        with self.assertRaisesRegex(ValueError, "stability limit"):
            _dark_timestep(static, dynamic)
        dynamic.update(Nx=1, dt=None)
        self.assertAlmostEqual(_dark_timestep(static, dynamic), .99*2/(1.3*20))
        static["dark_mu"] = 0.
        with self.assertRaisesRegex(ValueError, "Specify dt"):
            _dark_timestep(static, dynamic)
        dynamic["dt"] = .1
        self.assertEqual(_dark_timestep(static, dynamic), .1)


    def test_invalid_initial_field_shapes_and_types(self):
        s, d = parameters()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "field.npy")
            for values in (np.zeros((3, 1, 1)), np.full((32, 1, 1), np.nan),
                           np.ones((32, 1, 1), dtype=complex)):
                np.save(path, values)
                with self.assertRaises(ValueError):
                    load_dark_fields_from_toml({"dark_field0": dict(type=0, path=path)}, s, d)
            np.save(path, np.zeros((32, 1, 1)))
            for field_type in (-1, 7, True, 1.5):
                with self.assertRaises(ValueError):
                    load_dark_fields_from_toml({"dark_field0": dict(type=field_type, path=path)}, s, d)
            with self.assertRaisesRegex(ValueError, "evolved"):
                load_dark_fields_from_toml({"dark_field0": dict(type=0, path=path, evolve=False)}, s, d)
        with self.assertRaisesRegex(ValueError, "shape"):
            initialize_dark_photon_fields(s, d, phi=jnp.zeros((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "three components"):
            initialize_dark_photon_fields(s, d, E=(jnp.zeros(1),))



if __name__ == "__main__":
    unittest.main()
