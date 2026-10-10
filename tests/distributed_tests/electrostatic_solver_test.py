"""Focused cross-device regression tests."""

from tests.support.electrostatic_solver_fixtures import (
    TiledLocalSchwarzFixtures,
    _periodic_mode_problem,
    _relative_phi_error,
    _tile_field,
    jnp,
    solve_poisson_with_tiled_local_schwarz,
    unittest,
    update_tiled_ghost_cells,
)


class TestTiledLocalSchwarz(TiledLocalSchwarzFixtures, unittest.TestCase):
    def test_two_tiles_couple_through_the_interface(self):
        self._require_devices(2)
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, phi_true = _periodic_mode_problem(
            (2, 1, 1)
        )

        phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        local_cg_residual, schwarz_residual, schwarz_iteration = diagnostics

        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, 1)), 2.0e-5)
        self.assertEqual(local_cg_residual.shape, (2, 1, 1))
        self.assertTrue(jnp.all(jnp.isfinite(local_cg_residual)))
        self.assertLessEqual(float(schwarz_residual), 1.0e-6)
        self.assertGreater(int(schwarz_iteration), 0)


    def test_eight_tile_convergence_and_warm_start_behavior(self):
        self._require_devices(8)
        static_parameters, dynamic_parameters, rho, rho_tiles, phi_zero, phi_true = _periodic_mode_problem(
            (2, 2, 2)
        )

        phi_first, first_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_zero,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        self.assertLess(float(_relative_phi_error(phi_first, phi_true, 1)), 1.0e-3)
        self.assertLessEqual(float(first_diagnostics[1]), 1.0e-6)

        phi_warm, warm_converged_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_first,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        self.assertEqual(int(warm_converged_diagnostics[2]), 0)
        self.assertTrue(jnp.allclose(phi_warm, phi_first))

        rho_shifted = jnp.roll(rho, 1, axis=0)
        rho_shifted_tiles = _tile_field(rho_shifted, (2, 2, 2), (4, 4, 4), 1)
        rho_shifted_tiles = update_tiled_ghost_cells(
            rho_shifted_tiles,
            static_parameters,
            1,
        )
        _, warm_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_shifted_tiles,
            phi_first,
            static_parameters,
            dynamic_parameters,
            schwarz_max_iterations=0,
            return_diagnostics=True,
        )
        _, cold_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_shifted_tiles,
            phi_zero,
            static_parameters,
            dynamic_parameters,
            schwarz_max_iterations=0,
            return_diagnostics=True,
        )

        warm_defect = jnp.max(warm_diagnostics[0])
        cold_defect = jnp.max(cold_diagnostics[0])
        self.assertLess(float(warm_defect), 0.5 * float(cold_defect))


if __name__ == "__main__":
    unittest.main()
