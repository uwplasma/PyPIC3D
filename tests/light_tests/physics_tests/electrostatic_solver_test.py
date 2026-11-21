"""One-tile numerical and physical checks."""

from tests.support.electrostatic_solver_fixtures import (
    BC_CONDUCTING,
    TiledLocalSchwarzFixtures,
    _apply_tiled_phi_constant_boundaries,
    _assemble_owned,
    _periodic_mode_problem,
    _periodic_neutral_gaussian_problem,
    _poisson_residual,
    _relative_phi_error,
    _tile_field,
    _tiled_laplacian,
    jnp,
    kernel_parameters,
    solve_poisson_with_tiled_local_schwarz,
    unittest,
)


class TestTiledLocalSchwarz(TiledLocalSchwarzFixtures, unittest.TestCase):
    def test_one_tile_matches_manufactured_discrete_solution(self):
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, phi_true = _periodic_mode_problem(
            (1, 1, 1)
        )
        g = int(static_parameters.guard_cells)

        phi_tiles = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            schwarz_tol=1.0e-9,
            schwarz_max_iterations=1000,
            local_cg_tol=1.0e-9,
            local_cg_max_iterations=1000,
        )

        residual = _poisson_residual(
            rho_tiles,
            phi_tiles,
            dynamic_parameters,
            g,
        )
        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, g)), 2.0e-7)
        self.assertLess(float(jnp.max(jnp.abs(residual))), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
