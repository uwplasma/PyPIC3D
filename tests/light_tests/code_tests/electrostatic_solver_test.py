"""One-tile API contracts."""

from tests.support.electrostatic_solver_fixtures import (
    TiledLocalSchwarzFixtures,
    _periodic_mode_problem,
    jax,
    jnp,
    solve_poisson_with_tiled_local_schwarz,
    unittest,
)


class TestTiledLocalSchwarz(TiledLocalSchwarzFixtures, unittest.TestCase):
    def test_zero_residual_is_nan_safe_and_jittable(self):
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, _ = _periodic_mode_problem(
            (1, 1, 1)
        )
        rho_tiles = jnp.zeros_like(rho_tiles)
        phi_tiles = jnp.zeros_like(phi_tiles)

        solve = jax.jit(
            lambda rho, phi: solve_poisson_with_tiled_local_schwarz(
                rho,
                phi,
                static_parameters,
                dynamic_parameters,
                return_diagnostics=True,
            )
        )
        phi_tiles, diagnostics = solve(rho_tiles, phi_tiles)

        self.assertTrue(jnp.all(jnp.isfinite(phi_tiles)))
        self.assertTrue(jnp.allclose(phi_tiles, 0.0))
        self.assertTrue(jnp.allclose(diagnostics[0], 0.0))
        self.assertEqual(float(diagnostics[1]), 0.0)
        self.assertEqual(int(diagnostics[2]), 0)


if __name__ == "__main__":
    unittest.main()
