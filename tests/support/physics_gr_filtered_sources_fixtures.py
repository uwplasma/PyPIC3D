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


class FilteredSourcesTestsFixtures:
    def assertClose(self, actual, expected):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=2.e-12, atol=2.e-12)),
                        f"actual={actual}, expected={expected}")
