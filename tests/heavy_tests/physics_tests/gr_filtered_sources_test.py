from tests.support.physics_gr_filtered_sources_fixtures import (
    FilteredSourcesTestsFixtures,
    bssn_to_tiles,
    build_bssn_metric,
    jnp,
    parameters,
    particle_lorentz_factor,
    plasma,
    sample_energy,
    synchronize_particles,
    unittest,
)


class FilteredSourcesTests(FilteredSourcesTestsFixtures, unittest.TestCase):


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
