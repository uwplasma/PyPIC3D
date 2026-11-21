"""Focused cross-device regression tests."""

from tests.support.dark_boundaries_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    DarkBoundariesFixtures,
    advance_dark_pml,
    compute_dark_energy,
    dark_divergence,
    dark_field_boundaries,
    initialize_dark_photon_fields,
    initialize_dark_pml,
    interior,
    jax,
    jnp,
    kernel_parameters,
    load_pml_from_toml,
    np,
    pml_evolve,
    scalar_field_for_output,
    synchronized_dark_fields,
    unittest,
    wall_mode,
)


class TestDarkBoundaries(DarkBoundariesFixtures, unittest.TestCase):
    def test_pec_images_faces_corners_and_tile_seams(self):
        for g in (1, 2):
            for tiles in (1, 4):
                for bc in ((1, 1, 1), (1, 0, 1)):
                    s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, tile_shape=(8//tiles, 6, 4),
                                             guard_cells=g, boundary_conditions=bc, solver="dark_matter_yee")
                    for kind in ("E", "B", "phi"):
                        locations = (("C",)*3,) if kind == "phi" else B_FIELD_LOCATIONS if kind == "B" else D_FIELD_LOCATIONS
                        exact = []
                        for component, location in enumerate(locations):
                            parity = tuple(-1 if kind == "phi" else
                                           (-1 if axis == component else 1) if kind == "B" else
                                           (1 if axis == component else -1) for axis in range(3))
                            exact.append(wall_mode(s, d, location, parity))
                        values = tuple(jnp.zeros_like(v).at[interior(s)].set(v[interior(s)]) for v in exact)
                        result = dark_field_boundaries(values[0] if kind == "phi" else values, s, kind)
                        self.assert_tree_close(result, exact[0] if kind == "phi" else tuple(exact))


    def test_pml_jit_tiles_diagnostics_and_interior_constraint(self):
        results = []
        for tiles in (1, 4):
            s, d = kernel_parameters(Nx=32, Ny=1, Nz=1, tile_shape=(32//tiles, 1, 1),
                                     boundary_conditions=(1, 0, 0), dt=.01, dark_mu=.7,
                                     solver="dark_matter_yee", pml_active=True)
            E, A, phi = initialize_dark_photon_fields(s, d)
            ax = wall_mode(s, d, D_FIELD_LOCATIONS[0], (1, -1, -1))
            fields = initialize_dark_photon_fields(s, d, E, (ax, A[1], A[2]), phi)
            profiles = load_pml_from_toml(dict(wall="+x", thickness=8), s, d)[4]
            fields, state = initialize_dark_pml(fields, s, d, profiles)
            J = tuple(jnp.zeros_like(v) for v in E)
            self.assert_tree_close(advance_dark_pml(fields, J, state, s, d), pml_evolve(fields, state, s, d, 1))
            fields, state = pml_evolve(fields, state, s, d, 20)
            before = jax.tree.map(np.array, (fields, state))
            first = synchronized_dark_fields(fields, s, d, state)
            compute_dark_energy(fields, s, d, state)
            self.assert_tree_close(first, synchronized_dark_fields(fields, s, d, state))
            self.assert_tree_close(before, (fields, state), atol=0.)
            residual = dark_divergence(fields[0], s, d) + s.dark_mu**2*fields[2]
            # Divergence returns interior-only data; output assembly reads
            # overlapping halos at tile seams, so refresh the diagnostic first.
            residual = dark_field_boundaries(residual, s, "phi")
            residual = scalar_field_for_output(residual, s)[1:-1, 1:-1, 1:-1]
            np.testing.assert_allclose(residual[1:22], 0., atol=2e-13)
            results.append(jax.tree.map(lambda v: scalar_field_for_output(v, s)[1:-1, 1:-1, 1:-1], first))
        self.assertEqual(len(results), 2)
        self.assert_tree_close(*results)


if __name__ == "__main__":
    unittest.main()
