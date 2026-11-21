from tests.support.densitized_maxwell_runtime_fixtures import (
    signed_volume_metric,

    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    SpeciesConfig,
    TiledParticles,
    apply_tiled_supergaussian_absorber,
    build_field_output_map,
    build_yee_metric,
    coupled_setup,
    densitize_fields,
    densitize_vector,
    field_map_for_output,
    hybrid_boris_geodesic_push,
    jnp,
    np,
    owned_nodes,
    patch,
    physical_fields,
    physical_vector,
    random_vectors,
    reconstruct_vector,
    refresh_fields,
    time_loop_static_metric,
    unittest,
    update_B,
    update_D,
)


class TestDensitizedMaxwellFixtures:
    def assert_vectors_close(self, actual, expected, tolerance=3e-13):
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
