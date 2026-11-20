from tests.support.gr_work_energy_theorem_runtime_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    EnergySample,
    SpeciesConfig,
    TiledParticles,
    WorkEnergyAssertions,
    advance_tracker,
    build_bssn_metric,
    build_bssn_parameters,
    jax,
    jnp,
    native_runtime,
    parameters,
    sample_energy,
    start_tracker,
    surface_integral,
    tracker_values,
    unittest,
    volume_integral,
)


class WorkEnergyArithmeticTests(WorkEnergyAssertions):
    def test_linear_power_trapezoid_and_signed_step_residual(self):
        initial = EnergySample(*map(jnp.asarray, (0., 10., 2., 1., -.5, .2, 0.)))
        tracker = start_tracker(initial, reference=3.)
        # Integrated geometric work is 3 - 0.5; outward transport is 0.6.
        final = EnergySample(*map(jnp.asarray, (2., 12.4, 2., 2., 0., .4, 0.)))
        tracker = jax.jit(advance_tracker)(tracker, final)
        self.assertClose(tracker.curvature_work, 3.)
        self.assertClose(tracker.lapse_work, -.5)
        self.assertClose(tracker.boundary_work, .6)
        self.assertClose(tracker.step_residual, .5)
        self.assertClose(tracker.reference, 3.)
        self.assertEqual(int(tracker.steps), 1)

        loss = final._replace(time=jnp.asarray(3.), electromagnetic=jnp.asarray(10.))
        tracker = advance_tracker(tracker, loss)
        self.assertLess(float(tracker.step_residual), 0.)

    def test_full_domain_volume_and_face_quadrature_are_exact_for_linear_data(self):
        static, dynamic = parameters(
            Nx=8, Ny=4, Nz=1, x_wind=4., y_wind=2., z_wind=.7,
            x_min=.2, y_min=-.3, tile_shape=(8, 4, 1),
            boundary_conditions=(BC_CONSTANT, BC_CONDUCTING, 0),
        )
        x = .2 + jnp.arange(8)[:, None, None] * .5
        y = -.3 + jnp.arange(5)[None, :, None] * .5
        one = jnp.ones((8, 5, 1))
        integrand = one + 2*x + 3*y
        volume = 4 * 2 * .7
        self.assertClose(volume_integral(integrand, static, dynamic),
                         volume * (1 + 2*(.2 + 2.) + 3*(-.3 + 1.)))
        flux = jnp.stack((2*x*one, -.4*y*one, jnp.zeros_like(one)))
        self.assertClose(surface_integral(flux, static, dynamic), 1.6 * volume)

    def test_periodic_face_pairs_cancel(self):
        static, dynamic = parameters()
        flux = jnp.arange(24.).reshape(3, 8, 1, 1)
        self.assertClose(surface_integral(flux, static, dynamic), 0.)

    def test_absorbing_boundary_work_does_not_erase_continuum_residual(self):
        initial = EnergySample(*map(jnp.asarray, (0., 5., 0., 0., 0., 0., -2.)))
        final = initial._replace(time=jnp.asarray(1.), electromagnetic=jnp.asarray(3.))
        tracker = advance_tracker(start_tracker(initial), final)
        self.assertClose(tracker.absorbing_boundary_work, -2.)
        self.assertClose(tracker.step_residual, -2.)
        self.assertClose(tracker.reference, 5.)
        values = tracker_values(tracker)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(value))) for value in values.values()))


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):
    def test_offdiagonal_metric_and_curvature_contractions(self):
        static, dynamic = parameters()
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        curvature = jnp.asarray(((.03, -.02, .01), (-.02, -.04, .015), (.01, .015, .02)))
        particles, species, fields, static, dynamic = native_runtime(
            static, dynamic, gamma=gamma, curvature=curvature)
        sample = jax.jit(lambda p, f, d: sample_energy(p, species, f, static, d))(
            particles, fields, dynamic)
        electric, magnetic = jnp.asarray((.5, -.3, .2)), jnp.asarray((.1, .4, -.2))
        e_cov, b_cov = gamma @ electric, gamma @ magnetic
        rho = (electric @ e_cov + magnetic @ b_cov) / (8*jnp.pi)
        stress = rho * gamma - (jnp.outer(e_cov, e_cov) + jnp.outer(b_cov, b_cov)) / (4*jnp.pi)
        inverse = jnp.linalg.inv(gamma)
        proper_volume = 8 * jnp.sqrt(jnp.linalg.det(gamma))
        self.assertClose(sample.electromagnetic, rho * proper_volume)
        self.assertClose(sample.curvature_power,
                         jnp.sum(curvature * (inverse @ stress @ inverse)) * proper_volume)
        self.assertClose(sample.lapse_power, 0.)
        self.assertClose(sample.boundary_power, 0.)

    def test_lapse_power_uses_raised_momentum_and_negative_sign(self):
        static, dynamic = parameters(Nx=16, x_wind=8., tile_shape=(16, 1, 1))
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        particles, species, fields, static, dynamic = native_runtime(static, dynamic, gamma=gamma)
        wave_number = 2*jnp.pi / 8.
        x = jnp.arange(16)[:, None, None] * dynamic.dx
        bssn = dynamic.bssn_parameters
        state = bssn.state._replace(lapse=1 + .2*jnp.sin(wave_number*x))
        dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))
        metric = build_bssn_metric(state, static, dynamic)
        tiled_x = dynamic.grids.tiled_center_grid[0][..., :, None, None]
        zero = tuple(jnp.zeros_like(value) for value in fields[0])
        electric = (zero[0], .7*jnp.cos(wave_number*tiled_x)*metric.D[1].sqrt_gamma, zero[2])
        magnetic = (zero[0], zero[1], .3*metric.B[2].sqrt_gamma)
        history = (electric, magnetic, magnetic, fields[7][3], None)
        fields = (electric, magnetic, zero, fields[3], fields[4], fields[5], metric, history, fields[8])
        sample = sample_energy(particles, species, fields, static, dynamic)
        theta = wave_number * dynamic.dx
        discrete_k = (8*jnp.sin(theta) - jnp.sin(2*theta)) / (6*dynamic.dx)
        expected = (-8 * jnp.linalg.det(gamma) * jnp.linalg.inv(gamma)[0, 0]
                    * .7 * .3 * .2 * discrete_k / (8*jnp.pi))
        self.assertClose(sample.lapse_power, expected)
        self.assertLess(float(sample.lapse_power), 0.)
        self.assertClose(sample.curvature_power, 0.)

    def test_native_absorber_reports_signed_damping_and_reference_power(self):
        static, dynamic = parameters()
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        initial = sample_energy(particles, species, fields, static, dynamic)
        sigma = .12
        for reference_factor in (0., 2.):
            with self.subTest(reference_factor=reference_factor):
                absorber = tuple((tuple(jnp.full_like(value, sigma) for value in vector),
                                  tuple(reference_factor*value for value in vector),
                                  tuple(jnp.zeros_like(value, dtype=bool) for value in vector))
                                 for vector in (fields[0], fields[7][2]))
                history = fields[7][:4] + (absorber,)
                damped = fields[:7] + (history, fields[8])
                sample = sample_energy(particles, species, damped, static, dynamic)
                self.assertClose(sample.absorbing_boundary_power,
                                 2*sigma*(reference_factor-1)*initial.electromagnetic)

    def test_supergaussian_power_uses_native_damping_profile(self):
        static, dynamic = parameters(
            supergaussian_active=True, supergaussian_layers=((0, -1, 8, 2., .12),))
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        sample = sample_energy(particles, species, fields, static, dynamic)
        mean_sigma = jnp.mean(.12*((8-jnp.arange(8))/8)**2)
        self.assertClose(sample.absorbing_boundary_power, -2*mean_sigma*sample.electromagnetic)

    def test_particle_energy_includes_rest_mass_weight_and_offdiagonal_metric(self):
        static, dynamic = parameters()
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        _, _, fields, static, dynamic = native_runtime(static, dynamic, gamma=gamma)
        x = jnp.asarray((1.25, 0., 0.)).reshape(1, 1, 1, 1, 1, 3)
        u = jnp.asarray((.3, -.2, .1)).reshape(x.shape)
        particles = TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool))
        species = SpeciesConfig(jnp.asarray((0.,)), jnp.asarray((2.,)), jnp.asarray((.3,)),
                                jnp.ones((1, 3), dtype=bool))
        sample = sample_energy(particles, species, fields, static, dynamic)
        momentum = u.reshape(3)
        gamma_particle = jnp.sqrt(1 + momentum @ jnp.linalg.inv(gamma) @ momentum)
        self.assertClose(sample.particles, .6*gamma_particle)
        self.assertGreater(float(sample.particles), .6)

    def test_prescribed_flrw_volume_work_and_second_order_budget(self):
        static, dynamic = parameters()
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        initial_state = dynamic.bssn_parameters.state
        hubble = .3

        def at_time(time):
            a = jnp.exp(hubble*time)
            state = initial_state._replace(
                conformal_factor=jnp.ones_like(initial_state.conformal_factor)/a,
                trace_K=jnp.full_like(initial_state.trace_K, -3*hubble),
            )
            parameters_at_time = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(
                state=state, time=time))
            metric = build_bssn_metric(state, static, parameters_at_time)
            current_fields = fields[:6] + (metric,) + fields[7:]
            return sample_energy(particles, species, current_fields, static, parameters_at_time)

        initial = at_time(jnp.asarray(0.))
        last = at_time(jnp.asarray(1.))
        self.assertClose(last.electromagnetic, initial.electromagnetic*jnp.exp(-hubble))
        self.assertClose(last.curvature_power, -hubble*last.electromagnetic)
        self.assertClose(last.lapse_power, 0.)
        self.assertClose(last.boundary_power, 0.)

        def integrate(times):
            def accumulate(tracker, time):
                tracker = advance_tracker(tracker, at_time(time))
                return tracker, tracker.step_residual
            return jax.lax.scan(accumulate, start_tracker(initial), times)[0]

        residuals = []
        for steps in (8, 16, 32):
            tracker = jax.jit(integrate)(jnp.arange(1, steps+1, dtype=jnp.float64)/steps)
            residual = (tracker.latest.electromagnetic - tracker.baseline.electromagnetic
                        - tracker.curvature_work)
            self.assertGreater(float(residual), 0.)
            residuals.append(residual)
        self.assertClose(residuals[0]/residuals[1], 4., rtol=2.e-4)
        self.assertClose(residuals[1]/residuals[2], 4., rtol=2.e-4)

    def test_uniform_native_energy_excludes_all_halo_widths(self):
        samples = []
        for guard in (3, 5):
            static, dynamic = parameters(Nx=16, x_wind=16., tile_shape=(16, 1, 1), guard_cells=guard)
            particles, species, fields, static, dynamic = native_runtime(static, dynamic)
            samples.append(sample_energy(particles, species, fields, static, dynamic))
        self.assertClose(samples[0].electromagnetic, samples[1].electromagnetic)


if __name__ == "__main__":
    unittest.main()
