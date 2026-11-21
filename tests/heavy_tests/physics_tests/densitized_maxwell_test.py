from tests.support.physics_densitized_maxwell_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    TestDensitizedMaxwellFixtures,
    coupled_setup,
    curl,
    densitize_vector,
    jnp,
    owned_nodes,
    physical_vector,
    random_vectors,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestDensitizedMaxwell(TestDensitizedMaxwellFixtures, unittest.TestCase):


    def test_updates_match_physical_equations_and_boundaries(self):
        # Non-orthogonal conducting edges on the complete physical domain.
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (1, 1, 0))
        D, B, J = random_vectors(m)
        dt = .007
        for magnetic, locations, samples, initial, auxiliary in (
                (False, D_FIELD_LOCATIONS, m.D, D, B),
                (True, B_FIELD_LOCATIONS, m.B, B, D)):
            derivatives = curl(auxiliary, d, magnetic)
            expected = tuple(jnp.where(owned_nodes(value.shape, locations[i], s),
                value + dt*((-1 if magnetic else 1)*derivatives[i]/samples[i].sqrt_gamma
                            - (0 if magnetic else 4*jnp.pi*J[i])), value)
                for i, value in enumerate(initial))
            expected = refresh_fields(densitize_vector(expected, samples), s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B(auxiliary, density, m, s, d, dt) if magnetic else
                      update_D(density, auxiliary, densitize_vector(J, m.D), m, s, d, dt))
            self.assert_vectors_close(physical_vector(actual, samples), physical_vector(expected, samples), 2e-12)


if __name__ == "__main__":
    unittest.main()
