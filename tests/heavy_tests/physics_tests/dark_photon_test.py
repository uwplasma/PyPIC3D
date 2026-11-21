"""One-tile numerical and physical checks."""

from tests.support.dark_photon_fixtures import (
    D_FIELD_LOCATIONS,
    DarkPhotonFixtures,
    build_tiled_particles,
    compute_dark_B,
    contextlib,
    dark_divergence,
    dark_gradient,
    dark_photon_push_fields,
    evolve,
    filter_electric_field_for_particles,
    initialize_dark_photon_fields,
    initialize_fields,
    initialize_simulation,
    interior,
    io,
    jax,
    jnp,
    kernel_parameters,
    np,
    os,
    parameters,
    particle_species,
    synchronized_dark_fields,
    tempfile,
    time_loop_dark_photon,
    time_loop_electrodynamic,
    unittest,
    update_dark_E,
    update_tiled_ghost_cells,
    update_tiled_ghost_cells,
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
                s, d = parameters(n, dt=1/1024)
                state = initialize_dark_photon_fields(s, d, *wave(s, d, longitudinal))
                result = evolve(state, s, d, 1024)
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


    def test_yee_identities_and_gauss_constraint(self):
        for g in (1, 2):
            s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, guard_cells=g, dark_mu=.7, dt=.001)
            zero = initialize_fields(s, d)[3]
            noise = jnp.asarray(np.random.default_rng(42).normal(size=zero.shape))
            noise = update_tiled_ghost_cells(noise, s, g, location=("C", "C", "C"))
            E = dark_gradient(noise, s, d)
            E = update_tiled_ghost_cells(E, s, g, location=D_FIELD_LOCATIONS)
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


if __name__ == "__main__":
    unittest.main()
