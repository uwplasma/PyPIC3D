"""Focused cross-device regression tests."""

from tests.support.dark_photon_fixtures import (
    DarkPhotonFixtures,
    build_tiled_particles,
    compute_dark_B,
    compute_rho,
    evolve,
    initialize_dark_photon_fields,
    initialize_fields,
    jax,
    jnp,
    math,
    np,
    parameters,
    particle_species,
    scalar_field_for_output,
    time_loop_dark_photon,
    time_loop_electrodynamic,
    unittest,
    update_dark_A,
    update_dark_E,
    update_dark_phi,
    wave,
)


class TestDarkPhoton(DarkPhotonFixtures, unittest.TestCase):
    def test_eager_jit_and_four_tile_equivalence(self):
        if len(jax.devices()) < 4:
            raise RuntimeError("requires JAX_NUM_CPU_DEVICES=4 or more")
        results = []
        for tiles in (1, 4):
            s, d = parameters(tiles=tiles)
            state = initialize_dark_photon_fields(s, d, *wave(s, d, True))
            J = tuple(jnp.zeros_like(v) for v in state[0])
            E, A, phi = state
            A = update_dark_A(E, A, phi, s, d, d.dt)
            B = compute_dark_B(A, s, d)
            phi = update_dark_phi(A, phi, s, d, d.dt)
            E = update_dark_E(E, B, A, J, s, d, d.dt)
            self.assert_tree_close((E, A, phi), evolve(state, s, d, 1))
            result = evolve(state, s, d, 16)
            results.append(jax.tree.map(lambda v: scalar_field_for_output(v, s)[1:-1, 1:-1, 1:-1], result))
        self.assert_tree_close(*results)


    def test_shared_esirkepov_deposits_before_tile_and_periodic_crossings(self):
        s, d = parameters(n=8, tiles=2, sin_chi=0., current_deposition="esirkepov",
                          current_filter="none", relativistic=False)
        x = np.array([-.001, math.pi - .001])
        velocity = .2
        particles, species = build_tiled_particles([
            particle_species("p", charge=.01, mass=1., x1=x, u1=[velocity, velocity]),
        ], s, d)
        E, B, J, phi, rho = initialize_fields(s, d)
        fields = (E, B, J, rho, phi, (E, B), None)
        dark = initialize_dark_photon_fields(s, d)

        def physical(scalar):
            return np.asarray(scalar_field_for_output(scalar, s))[1:-1, 1:-1, 1:-1]

        rho_old = physical(compute_rho(particles, species, rho, s, d))
        for loop, state in (
            (time_loop_electrodynamic, fields + (jnp.asarray(False),)),
            (time_loop_dark_photon, fields + (dark, jnp.asarray(False))),
        ):
            with self.subTest(loop=loop.__name__):
                advanced, result = jax.jit(lambda p, f: loop(p, species, f, s, d))(particles, state)
                expected_x = (x + velocity * float(d.dt) + math.pi) % (2 * math.pi) - math.pi
                actual_x = np.asarray(advanced.x)[np.asarray(advanced.active), 0]
                np.testing.assert_allclose(np.sort(actual_x), np.sort(expected_x), atol=1e-14)
                self.assertFalse(bool(result[-1]))
                rho_new = physical(compute_rho(advanced, species, rho, s, d))
                current_x = physical(result[2][0])
                residual = (rho_new - rho_old) / float(d.dt) + (
                    current_x - np.roll(current_x, 1, axis=0)
                ) / float(d.dx)
                np.testing.assert_allclose(residual, 0., atol=1e-12)


if __name__ == "__main__":
    unittest.main()
