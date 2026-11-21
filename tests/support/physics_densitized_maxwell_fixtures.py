from tests.support.densitized_maxwell_runtime_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Esirkepov_current,
    SpeciesConfig,
    TiledParticles,
    compute_covariant_E,
    compute_covariant_H,
    compute_rho,
    coupled_setup,
    curl,
    densitize_vector,
    divergence,
    jnp,
    np,
    owned_nodes,
    physical_vector,
    random_vectors,
    reconstruct_vector,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestDensitizedMaxwellFixtures:
    def assert_vectors_close(self, actual, expected, tolerance=3e-13):
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
