from tests.support.code_densitized_maxwell_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    TestDensitizedMaxwellFixtures,
    apply_tiled_supergaussian_absorber,
    coupled_setup,
    densitize_vector,
    jnp,
    physical_vector,
    random_vectors,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestDensitizedMaxwell(TestDensitizedMaxwellFixtures, unittest.TestCase):


    def test_horizon_and_absorber_keep_physical_boundary_policy(self):
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (3, 1, 0))
        s = s._replace(horizon_field_cells=2, supergaussian_active=True,
                       supergaussian_layers=((0, 1, 3, 4., 10.),))
        D, B, _ = random_vectors(m)
        zero = tuple(jnp.zeros_like(a) for a in D)
        for magnetic, locations, samples, initial in (
                (False, D_FIELD_LOCATIONS, m.D, D),
                (True, B_FIELD_LOCATIONS, m.B, B)):
            # the absorber damps the physical field, then the boundaries refresh its density
            expected = apply_tiled_supergaussian_absorber(initial, s, d, .03, locations=locations)
            expected = refresh_fields(densitize_vector(expected, samples), s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B(zero, density, m, s, d, .03) if magnetic else
                      update_D(density, zero, zero, m, s, d, .03))
            self.assert_vectors_close(physical_vector(actual, samples), physical_vector(expected, samples))


if __name__ == "__main__":
    unittest.main()
