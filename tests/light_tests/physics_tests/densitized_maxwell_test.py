from tests.support.physics_densitized_maxwell_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Esirkepov_current,
    SpeciesConfig,
    TestDensitizedMaxwellFixtures,
    TiledParticles,
    compute_covariant_E,
    compute_covariant_H,
    compute_rho,
    coupled_setup,
    densitize_vector,
    divergence,
    jnp,
    np,
    random_vectors,
    reconstruct_vector,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestDensitizedMaxwell(TestDensitizedMaxwellFixtures, unittest.TestCase):

    def test_constitutive_fields_match_physical_reconstruction(self):
        _, _, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        density_D, density_B = densitize_vector(D, m.D), densitize_vector(B, m.B)

        def upstream(value, location, axis, beta):
            # Independent three-point half-node weights for the shift term.
            def neighbor(offset):
                indices = (jnp.arange(value.shape[axis+3])+offset) % value.shape[axis+3]
                return jnp.take(value, indices, axis=axis+3)
            if location[axis] == 'C':
                left = -.25*neighbor(-1)+value+.25*neighbor(1)
                right = .25*value+neighbor(1)-.25*neighbor(2)
            else:
                left = -.25*neighbor(-2)+neighbor(-1)+.25*value
                right = .25*neighbor(-1)+value-.25*neighbor(1)
            return beta*jnp.where(beta > 0, right, left)

        for magnetic, locations, samples, compute in (
                (False, D_FIELD_LOCATIONS, m.D, compute_covariant_E),
                (True, B_FIELD_LOCATIONS, m.B, compute_covariant_H)):
            own, other = (density_B, density_D) if magnetic else (density_D, density_B)
            other_locations = D_FIELD_LOCATIONS if magnetic else B_FIELD_LOCATIONS
            sign = -1 if magnetic else 1
            expected = []
            for i, (location, sample) in enumerate(zip(locations, samples)):
                vector = jnp.stack(reconstruct_vector(own, locations, location), axis=-1)
                lower = jnp.sum(sample.gamma[..., i, :]*vector/sample.sqrt_gamma[..., None], axis=-1)
                j, k = (i+1) % 3, (i+2) % 3
                cross = (upstream(other[k], other_locations[k], j, sample.shift[..., j])
                         - upstream(other[j], other_locations[j], k, sample.shift[..., k]))
                expected.append(sample.lapse*lower+sign*cross)
            actual = compute(density_D, density_B, m)
            self.assertTrue(bool(jnp.allclose(jnp.stack(actual), jnp.stack(expected),
                                             rtol=3e-13, atol=3e-13)))


    def test_vacuum_curls_preserve_density_divergences(self):
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        D = refresh_fields(densitize_vector(D, m.D), s, D_FIELD_LOCATIONS, 'D', m)
        B = refresh_fields(densitize_vector(B, m.B), s, B_FIELD_LOCATIONS, 'B', m)
        # Auxiliary fields must also have periodic, communicated halos.
        auxiliary = refresh_fields(random_vectors(m)[2], s, D_FIELD_LOCATIONS)
        zero = tuple(jnp.zeros_like(a) for a in D)
        next_D = update_D(D, auxiliary, zero, m, s, d, .001)
        auxiliary_B = refresh_fields(random_vectors(m)[2], s, B_FIELD_LOCATIONS)
        next_B = update_B(auxiliary_B, B, m, s, d, .001)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        for before, after, forward in ((D, next_D, False), (B, next_B, True)):
            error = divergence(after, d, forward=forward)-divergence(before, d, forward=forward)
            np.testing.assert_allclose(error[interior], 0., rtol=0, atol=1e-11)

    def test_esirkepov_source_preserves_gauss_in_curved_metric(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        s = s._replace(current_deposition='esirkepov', current_filter='none')
        d = d._replace(dt=.02)
        x = jnp.array([.43, .47, .5]).reshape(1, 1, 1, 1, 1, 3)
        old = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], dtype=bool))
        new = old._replace(x=x+jnp.array([.01, -.02, 0.]))
        species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1),
                                jnp.array([[True, True, False]]))
        scalar = jnp.zeros_like(m.center.sqrt_gamma)
        zero = (scalar,)*3
        current = Esirkepov_current(old, new, species, zero, s, d)
        updated = update_D(zero, zero, current, m, s, d, d.dt)
        rho_old = compute_rho(old, species, scalar, s, d)
        rho_new = compute_rho(new, species, scalar, s, d)
        drift = divergence(updated, d)-4*jnp.pi*(rho_new-rho_old)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        np.testing.assert_allclose(drift[interior], 0., rtol=0., atol=1e-11)


if __name__ == "__main__":
    unittest.main()
