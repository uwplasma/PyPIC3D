"""Single-device numerical tests."""

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
    jax,
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


    def test_periodic_neutral_gaussian_is_second_order_in_the_interior(self):

        resolutions = (16, 24, 32)
        schwarz_tol = 1.0e-8
        schwarz_max_iterations = 600
        spacings = []
        potential_errors = []

        for cells_per_axis in resolutions:
            (
                static_parameters,
                dynamic_parameters,
                rho,
                rho_tiles,
                phi_zero,
                phi_true,
                comparison_mask,
            ) = _periodic_neutral_gaussian_problem(cells_per_axis)

            cell_volume = (
                dynamic_parameters.dx
                * dynamic_parameters.dy
                * dynamic_parameters.dz
            )
            total_grid_charge = jnp.sum(rho) * cell_volume
            self.assertLess(float(jnp.abs(total_grid_charge)), 1.0e-12)

            phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
                rho_tiles,
                phi_zero,
                static_parameters,
                dynamic_parameters,
                schwarz_tol=schwarz_tol,
                schwarz_max_iterations=schwarz_max_iterations,
                local_cg_tol=1.0e-10,
                local_cg_max_iterations=1000,
                return_diagnostics=True,
            )
            residual = _poisson_residual(
                rho_tiles,
                phi_tiles,
                dynamic_parameters,
                1,
            )

            self.assertLessEqual(
                float(jnp.max(jnp.abs(residual))),
                schwarz_tol,
            )
            self.assertLess(int(diagnostics[2]), schwarz_max_iterations)

            phi = _assemble_owned(phi_tiles, 1)
            gauge_offset = jnp.mean((phi - phi_true)[comparison_mask])
            potential_error = phi - phi_true - gauge_offset
            l2_error = jnp.sqrt(
                jnp.mean(potential_error[comparison_mask] ** 2)
            )

            spacings.append(float(dynamic_parameters.dx))
            potential_errors.append(float(l2_error))

        for coarse_error, fine_error in zip(
            potential_errors[:-1],
            potential_errors[1:],
        ):
            self.assertLess(fine_error, coarse_error)

        for coarse_index in range(len(resolutions) - 1):
            order = jnp.log(
                potential_errors[coarse_index]
                / potential_errors[coarse_index + 1]
            ) / jnp.log(
                spacings[coarse_index]
                / spacings[coarse_index + 1]
            )
            self.assertAlmostEqual(float(order), 2.0, delta=0.2)


    def test_conducting_boundaries_preserve_the_existing_constant_ghost_rule(self):
        Nx = Ny = Nz = 8
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            tile_shape=(Nx, Ny, Nz),
            guard_cells=1,
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
            electrostatic=True,
            solver="electrostatic",
        )
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx),
            jnp.arange(Ny),
            jnp.arange(Nz),
            indexing="ij",
        )
        phi_true = (
            jnp.cos(jnp.pi * ii / (Nx - 1))
            + 0.2 * jnp.cos(2.0 * jnp.pi * jj / (Ny - 1))
            + 0.1 * jnp.cos(jnp.pi * kk / (Nz - 1))
        )
        phi_true_tiles = _tile_field(phi_true, (1, 1, 1), (Nx, Ny, Nz), 1)
        phi_true_tiles = _apply_tiled_phi_constant_boundaries(
            phi_true_tiles,
            static_parameters,
            1,
        )
        rho_owned = -dynamic_parameters.eps * _tiled_laplacian(
            phi_true_tiles,
            dynamic_parameters,
            1,
        )
        rho_tiles = jnp.zeros_like(phi_true_tiles)
        rho_tiles = rho_tiles.at[..., 1:-1, 1:-1, 1:-1].set(rho_owned)

        phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            jnp.zeros_like(phi_true_tiles),
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        local_cg_residual, schwarz_residual, schwarz_iteration = diagnostics

        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, 1)), 2.0e-4)
        self.assertTrue(jnp.all(jnp.isfinite(local_cg_residual)))
        self.assertLessEqual(float(schwarz_residual), 1.0e-6)
        self.assertGreater(int(schwarz_iteration), 0)
        self.assertTrue(
            jnp.allclose(
                phi_tiles[..., 0, 1:-1, 1:-1],
                phi_tiles[..., 1, 1:-1, 1:-1],
            )
        )
        self.assertTrue(
            jnp.allclose(
                phi_tiles[..., -1, 1:-1, 1:-1],
                phi_tiles[..., -2, 1:-1, 1:-1],
            )
        )


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
