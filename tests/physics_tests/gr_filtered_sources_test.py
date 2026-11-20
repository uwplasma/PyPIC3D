from tests.support.gr_filtered_sources_runtime_fixtures import (
    compute_electromagnetic_energy_momentum,
    particle_lorentz_factor,
    sample_energy,
    synchronize_particles,

    binomial,
    bssn_to_tiles,
    build_bssn_metric,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    electromagnetic_moments,
    jnp,
    parameters,
    plasma,
    unittest,
)


class FilteredSourcesTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=2.e-12, atol=2.e-12)),
                        f"actual={actual}, expected={expected}")

    def test_particle_gamma_uses_filtered_metric_but_raw_destination_volume(self):
        static, dynamic = parameters()
        particles, species = plasma(dynamic)
        state = dynamic.bssn_parameters.state
        metric = build_bssn_metric(state, static, dynamic)
        rho, momentum, stress = compute_particle_energy_momentum(
            particles, species, metric, static, dynamic)
        u = particles.u.reshape(3)
        Gamma = jnp.sqrt(1. + jnp.sum(u*u)/binomial(state.conformal_factor**-2)[2, 0, 0])
        mass = species.mass[0]*species.weight[0]
        volume = state.conformal_factor**-3
        self.assertClose(jnp.sum(volume*rho), mass*Gamma)
        self.assertClose(jnp.sum(volume*momentum, axis=(1, 2, 3)), mass*u)
        self.assertClose(jnp.sum(volume*stress, axis=(2, 3, 4)), mass*jnp.outer(u, u)/Gamma)
        self.assertEqual(int(jnp.count_nonzero(rho)), 1)

    def test_em_only_filter_keeps_checkerboard_energy(self):
        static, dynamic = parameters()
        particles, species = plasma(dynamic, empty=True)
        state = dynamic.bssn_parameters.state
        state = state._replace(conformal_factor=jnp.ones_like(state.conformal_factor))
        alternating = ((-1.)**jnp.arange(8))[:, None, None]
        density = bssn_to_tiles(alternating, static)
        zero = jnp.zeros_like(density)
        moments = electromagnetic_moments((density, zero, zero), (zero, zero, zero), static)
        rho, momentum, stress = compute_energy_momentum_tensor(
            particles, species, moments, state, static, dynamic)
        self.assertClose(rho, 1/(8*jnp.pi))
        self.assertClose(momentum, 0.)
        self.assertClose(stress[0, 0], -rho)
        self.assertClose(stress[1, 1], rho)
        self.assertClose(binomial(alternating), 0.)


    def test_total_sources_filter_density_once_at_each_stage(self):
        static, dynamic = parameters()
        particles, species = plasma(dynamic)
        shape = dynamic.bssn_parameters.state.conformal_factor.shape
        phase = 2*jnp.pi*jnp.arange(8)[:, None, None]/8
        d = jnp.stack((.2 + .1*jnp.cos(phase), .3*jnp.sin(phase), jnp.zeros(shape)))
        b = jnp.stack((jnp.zeros(shape), .1*jnp.ones(shape), .2*jnp.cos(phase)))
        moments = (jnp.einsum("i...,j...->ij...", d, d),
                   jnp.einsum("i...,j...->ij...", b, b),
                   jnp.einsum("i...,j...->ij...", d, b))
        u = particles.u.reshape(3)
        for scale in (1., 1.2):
            with self.subTest(stage_scale=scale):
                state = dynamic.bssn_parameters.state
                state = state._replace(conformal_factor=scale*state.conformal_factor)
                W = state.conformal_factor
                volume = W**-3
                # The particle sits exactly on C node 2: Hermite sampling of
                # the filtered physical metric equals that node's value.
                gamma_particle = binomial(W**-2)[2, 0, 0]
                Gamma = jnp.sqrt(1. + jnp.sum(u*u)/gamma_particle)
                mass = species.mass[0]*species.weight[0]
                cloud = jnp.zeros(shape).at[2, 0, 0].set(mass)
                pic = (cloud*Gamma/volume,
                       u[:, None, None, None]*cloud/volume,
                       jnp.outer(u, u)[:, :, None, None, None]*cloud/(Gamma*volume))
                em = compute_electromagnetic_energy_momentum(moments, state)
                raw = tuple(a+b for a, b in zip(pic, em))
                actual = compute_energy_momentum_tensor(
                    particles, species, moments, state, static, dynamic)
                for value, reference in zip(actual, raw):
                    self.assertClose(value, binomial(volume*reference)/volume)
                    self.assertClose(jnp.sum(value*volume, axis=(-3, -2, -1)),
                                     jnp.sum(reference*volume, axis=(-3, -2, -1)))
                self.assertClose(actual[2], jnp.swapaxes(actual[2], 0, 1))
                self.assertGreater(float(jnp.max(jnp.abs(actual[0]-binomial(raw[0])))), 1.e-4)

    def test_energy_measurement_is_raw_after_filtered_synchronization(self):
        static, dynamic = parameters()
        particles, species = plasma(dynamic)
        state = dynamic.bssn_parameters.state
        metric = build_bssn_metric(state, static, dynamic)
        phase = 2*jnp.pi*jnp.arange(8)[:, None, None]/8
        density = bssn_to_tiles(.4*jnp.cos(phase), static)
        zero = tuple(jnp.zeros_like(density) for _ in range(3))
        D, B = (density, density*.3, density*-.2), zero
        fields = (D, B, zero, zero[0], zero[0], (zero, zero), metric,
                  (D, B, B, jnp.asarray(.4), None), jnp.asarray(False))
        sample = sample_energy(particles, species, fields, static, dynamic)
        common = synchronize_particles(particles, species, fields, static, dynamic)
        raw_static = static._replace(current_filter="none")
        Gamma = particle_lorentz_factor(common, metric, raw_static, dynamic)
        mass = species.mass[0]*species.weight[0]
        self.assertClose(sample.particles, mass*jnp.sum(Gamma))
        wrong_common = synchronize_particles(particles, species, fields, raw_static, dynamic)
        wrong_Gamma = particle_lorentz_factor(wrong_common, metric, raw_static, dynamic)
        self.assertGreater(float(jnp.abs(sample.particles-mass*jnp.sum(wrong_Gamma))), 1.e-8)

        # With common-time momenta already supplied (zero synchronization
        # offset), changing the force filter must not redefine physical energy
        # or the original continuum geometry powers.
        common_fields = fields[:7] + ((D, B, B, jnp.asarray(0.), None), fields[8])
        filtered = sample_energy(common, species, common_fields, static, dynamic)
        raw = sample_energy(common, species, common_fields, raw_static, dynamic)
        for value, reference in zip(filtered, raw):
            self.assertClose(value, reference)

if __name__ == "__main__":
    unittest.main()
