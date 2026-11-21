"""One-tile numerical and physical checks."""

from tests.support.dark_boundaries_fixtures import (
    DarkBoundariesFixtures,
    evolve,
    initialize_dark_photon_fields,
    initialize_dark_pml,
    jnp,
    kernel_parameters,
    load_pml_from_toml,
    np,
    pml_evolve,
    stretch_dark_derivatives,
    synchronized_dark_fields,
    unittest,
)


class TestDarkBoundaries(DarkBoundariesFixtures, unittest.TestCase):

    def test_zero_sigma_matches_potential_solver_and_initial_reconstruction(self):
        for bc in ((0, 0, 0), (1, 0, 1)):
            s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, dt=.002, dark_mu=.7,
                                     boundary_conditions=bc, solver="dark_matter_yee")
            zero = initialize_dark_photon_fields(s, d)[2]
            rng = np.random.default_rng(93)
            E, A = [tuple(jnp.asarray(rng.normal(size=zero.shape)) for _ in range(3)) for _ in range(2)]
            fields = initialize_dark_photon_fields(s, d, E, A, jnp.asarray(rng.normal(size=zero.shape)))
            for strength in (0., 3.):
                profiles = load_pml_from_toml(dict(wall="+x", thickness=3, sigma_max=strength), s, d)[4]
                seeded, state = initialize_dark_pml(fields, s, d, profiles)
                expected = synchronized_dark_fields(fields, s, d)
                self.assert_tree_close(synchronized_dark_fields(seeded, s, d, state), expected)
                if strength == 0.:
                    result, _ = pml_evolve(seeded, state, s, d, 8)
                    self.assert_tree_close(result, evolve(fields, s, d, 8))


if __name__ == "__main__":
    unittest.main()
