from tests.support.gr_work_energy_theorem_runtime_fixtures import (
    EnergySample,
    SpeciesConfig,
    TiledParticles,
    WorkEnergyAssertions,
    build_bssn_parameters,
    build_static_parameters,
    dynamic_parameters_for_output,
    initialize_tracker,
    jnp,
    native_runtime,
    parameters,
    sample_energy,
    start_tracker,
    toml,
    unittest,
    update_parameters_from_toml,
)


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):


    def test_empty_energy_requires_positive_explicit_normalization(self):
        static, dynamic = parameters()
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        zero = tuple(jnp.zeros_like(value) for value in fields[0])
        fields = (zero, zero, zero) + fields[3:7] + ((zero, zero, zero, fields[7][3], None), fields[8])
        static = static._replace(gr_work_energy_theorem=True)
        with self.assertRaises(ValueError):
            initialize_tracker(particles, species, fields, static, dynamic)
        carry = initialize_tracker(
            particles, species, fields, static._replace(gr_work_energy_reference=2.), dynamic)
        self.assertClose(carry.reference, 2.)
        for reference in (0., -1.):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                initialize_tracker(
                    particles, species, fields, static._replace(gr_work_energy_reference=reference), dynamic)


if __name__ == "__main__":
    unittest.main()
