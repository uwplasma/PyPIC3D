import inspect
import unittest

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _apply_tiled_phi_zero_boundaries,
    _free_poisson_residual,
    _local_tile_cg_solve,
    _poisson_residual,
    _tiled_laplacian,
    calculate_electrostatic_fields,
    solve_poisson_with_tiled_local_schwarz,
)
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


def _tile_field(interior, tile_grid_shape, tile_shape, g):
    ntx, nty, ntz = tile_grid_shape
    tile_nx, tile_ny, tile_nz = tile_shape
    owned_tiles = interior.reshape(
        ntx,
        tile_nx,
        nty,
        tile_ny,
        ntz,
        tile_nz,
    ).transpose(0, 2, 4, 1, 3, 5)

    field_tiles = jnp.zeros(
        tile_grid_shape
        + (
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=interior.dtype,
    )
    return field_tiles.at[..., g:-g, g:-g, g:-g].set(owned_tiles)


def _assemble_owned(field_tiles, g):
    ntx, nty, ntz = field_tiles.shape[:3]
    owned_tiles = field_tiles[..., g:-g, g:-g, g:-g]
    tile_nx, tile_ny, tile_nz = owned_tiles.shape[-3:]
    return owned_tiles.transpose(0, 3, 1, 4, 2, 5).reshape(
        ntx * tile_nx,
        nty * tile_ny,
        ntz * tile_nz,
    )


def _periodic_mode_problem(tile_grid_shape, g=1):
    Nx = Ny = Nz = 8
    tile_shape = tuple(
        cells // num_tiles
        for cells, num_tiles in zip((Nx, Ny, Nz), tile_grid_shape)
    )
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        tile_shape=tile_shape,
        guard_cells=g,
        boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
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
        jnp.sin(2.0 * jnp.pi * ii / Nx)
        + 0.3 * jnp.cos(2.0 * jnp.pi * jj / Ny)
        + 0.2 * jnp.sin(2.0 * jnp.pi * kk / Nz)
    )
    negative_laplacian_phi = -(
        (
            jnp.roll(phi_true, 1, axis=0)
            + jnp.roll(phi_true, -1, axis=0)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dx**2
        + (
            jnp.roll(phi_true, 1, axis=1)
            + jnp.roll(phi_true, -1, axis=1)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dy**2
        + (
            jnp.roll(phi_true, 1, axis=2)
            + jnp.roll(phi_true, -1, axis=2)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dz**2
    )
    rho = dynamic_parameters.eps * negative_laplacian_phi
    rho_tiles = _tile_field(rho, tile_grid_shape, tile_shape, g)
    rho_tiles = update_tiled_ghost_cells(
        rho_tiles,
        static_parameters,
        g,
    )
    phi_tiles = _tile_field(
        jnp.zeros_like(phi_true),
        tile_grid_shape,
        tile_shape,
        g,
    )

    return static_parameters, dynamic_parameters, rho, rho_tiles, phi_tiles, phi_true


def _periodic_neutral_gaussian_problem(cells_per_axis, g=1):
    domain_width = 8.0
    sigma_inner = 0.75
    sigma_outer = 1.0
    comparison_radius = 2.0
    total_charge = 1.0
    eps = 1.0

    tile_grid_shape = (2, 2, 2)
    tile_shape = (cells_per_axis // 2,) * 3
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=cells_per_axis,
        Ny=cells_per_axis,
        Nz=cells_per_axis,
        x_wind=domain_width,
        y_wind=domain_width,
        z_wind=domain_width,
        tile_shape=tile_shape,
        guard_cells=g,
        boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        eps=eps,
        electrostatic=True,
        solver="electrostatic",
    )

    x = dynamic_parameters.grids.center[0][1:-1]
    y = dynamic_parameters.grids.center[1][1:-1]
    z = dynamic_parameters.grids.center[2][1:-1]
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    radius = jnp.sqrt(X * X + Y * Y + Z * Z)
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    cell_volume = (
        dynamic_parameters.dx
        * dynamic_parameters.dy
        * dynamic_parameters.dz
    )

    gaussian_normalization = (2.0 * jnp.pi) ** 1.5
    rho_inner = jnp.exp(
        -radius * radius / (2.0 * sigma_inner**2)
    ) / (
        gaussian_normalization * sigma_inner**3
    )
    rho_inner = rho_inner * total_charge / (
        jnp.sum(rho_inner) * cell_volume
    )
    # normalize the narrow Gaussian to exactly +Q on the sampled grid

    rho_outer = jnp.exp(
        -radius * radius / (2.0 * sigma_outer**2)
    ) / (
        gaussian_normalization * sigma_outer**3
    )
    rho_outer = rho_outer * (-total_charge) / (
        jnp.sum(rho_outer) * cell_volume
    )
    # normalize the broad Gaussian to exactly -Q for periodic neutrality

    rho = rho_inner + rho_outer


    rho_tiles = _tile_field(
        rho,
        tile_grid_shape,
        tile_shape,
        g,
    )
    rho_tiles = update_tiled_ghost_cells(
        rho_tiles,
        static_parameters,
        g,
    )
    phi_tiles = _tile_field(
        jnp.zeros_like(rho),
        tile_grid_shape,
        tile_shape,
        g,
    )

    phi_inner = total_charge * erf(
        radius / (jnp.sqrt(2.0) * sigma_inner)
    ) / (
        4.0 * jnp.pi * eps * safe_radius
    )
    phi_inner_at_origin = (
        total_charge
        * jnp.sqrt(2.0 / jnp.pi)
        / (4.0 * jnp.pi * eps * sigma_inner)
    )
    phi_inner = jnp.where(radius > 0.0, phi_inner, phi_inner_at_origin)

    phi_outer = total_charge * erf(
        radius / (jnp.sqrt(2.0) * sigma_outer)
    ) / (
        4.0 * jnp.pi * eps * safe_radius
    )
    phi_outer_at_origin = (
        total_charge
        * jnp.sqrt(2.0 / jnp.pi)
        / (4.0 * jnp.pi * eps * sigma_outer)
    )
    phi_outer = jnp.where(radius > 0.0, phi_outer, phi_outer_at_origin)

    phi_true = phi_inner - phi_outer
    comparison_mask = radius <= comparison_radius

    return (
        static_parameters,
        dynamic_parameters,
        rho,
        rho_tiles,
        phi_tiles,
        phi_true,
        comparison_mask,
    )


def _relative_phi_error(phi_tiles, phi_true, g):
    phi = _assemble_owned(phi_tiles, g)
    phi = phi - jnp.mean(phi)
    phi_true = phi_true - jnp.mean(phi_true)
    return jnp.linalg.norm(phi - phi_true) / jnp.linalg.norm(phi_true)


class TestTiledLocalSchwarz(unittest.TestCase):
    def _require_devices(self, count):
        if jax.device_count() < count:
            self.skipTest(f"Need {count} JAX devices, got {jax.device_count()}")

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

    def test_periodic_neutral_gaussian_is_second_order_in_the_interior(self):
        self._require_devices(8)

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

    def test_conducting_boundaries_ground_the_potential(self):
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
            jnp.sin(jnp.pi * ii / (Nx - 1))
            * jnp.sin(jnp.pi * jj / (Ny - 1))
            * jnp.sin(jnp.pi * kk / (Nz - 1))
        )
        phi_true_tiles = _tile_field(phi_true, (1, 1, 1), (Nx, Ny, Nz), 1)
        phi_true_tiles = _apply_tiled_phi_zero_boundaries(
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

        phi = _assemble_owned(phi_tiles, 1)
        relative_error = jnp.linalg.norm(phi - phi_true) / jnp.linalg.norm(phi_true)
        residual = _free_poisson_residual(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            1,
        )

        self.assertLess(float(relative_error), 2.0e-7)
        self.assertTrue(jnp.all(jnp.isfinite(local_cg_residual)))
        self.assertLessEqual(float(schwarz_residual), 1.0e-6)
        self.assertLessEqual(float(jnp.max(jnp.abs(residual))), 1.0e-6)
        self.assertGreater(int(schwarz_iteration), 0)
        self.assertTrue(jnp.all(phi[0, :, :] == 0.0))
        self.assertTrue(jnp.all(phi[-1, :, :] == 0.0))
        self.assertTrue(jnp.all(phi[:, 0, :] == 0.0))
        self.assertTrue(jnp.all(phi[:, -1, :] == 0.0))
        self.assertTrue(jnp.all(phi[:, :, 0] == 0.0))
        self.assertTrue(jnp.all(phi[:, :, -1] == 0.0))

    def _assert_non_neutral_grounded_solution(self, tile_nx):
        Nx = 16
        tile_grid_shape = (Nx // tile_nx, 1, 1)
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=Nx,
            Ny=1,
            Nz=1,
            tile_shape=(tile_nx, 1, 1),
            guard_cells=1,
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            electrostatic=True,
            solver="electrostatic",
        )
        i = jnp.arange(Nx, dtype=jnp.float64)
        phi_true = (i * (Nx - 1 - i))[:, jnp.newaxis, jnp.newaxis]
        phi_true_tiles = _tile_field(
            phi_true,
            tile_grid_shape,
            (tile_nx, 1, 1),
            1,
        )
        phi_true_tiles = _apply_tiled_phi_zero_boundaries(
            phi_true_tiles,
            static_parameters,
            1,
        )
        rho_owned = jnp.full(
            tile_grid_shape + (tile_nx, 1, 1),
            2.0 * dynamic_parameters.eps / dynamic_parameters.dx**2,
            dtype=phi_true.dtype,
        )
        rho_tiles = jnp.zeros_like(phi_true_tiles)
        rho_tiles = rho_tiles.at[..., 1:-1, 1:-1, 1:-1].set(rho_owned)

        phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            jnp.zeros_like(phi_true_tiles),
            static_parameters,
            dynamic_parameters,
            schwarz_tol=1.0e-9,
            local_cg_tol=1.0e-9,
            return_diagnostics=True,
        )
        phi = _assemble_owned(phi_tiles, 1)
        residual = _free_poisson_residual(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            1,
        )

        self.assertGreater(float(jnp.sum(rho_owned)), 0.0)
        self.assertLess(float(jnp.max(jnp.abs(phi - phi_true))), 1.0e-8)
        self.assertEqual(float(phi[0, 0, 0]), 0.0)
        self.assertEqual(float(phi[-1, 0, 0]), 0.0)
        self.assertLessEqual(float(jnp.max(jnp.abs(residual))), 1.0e-9)
        self.assertLess(int(diagnostics[2]), 500)

    def test_non_neutral_one_dimensional_source_converges_between_grounded_walls(self):
        self._assert_non_neutral_grounded_solution(tile_nx=16)

    def test_two_tiles_keep_only_the_global_conducting_walls_grounded(self):
        self._require_devices(2)
        self._assert_non_neutral_grounded_solution(tile_nx=8)

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

    def test_local_cg_reductions_keep_the_tile_axes(self):
        source = inspect.getsource(_local_tile_cg_solve)
        production_source = inspect.getsource(calculate_electrostatic_fields)

        self.assertEqual(source.count("axis=(-3, -2, -1)"), 3)
        self.assertIn("solve_poisson_with_tiled_local_schwarz", production_source)
        self.assertNotIn("conjugate_gradient", production_source)
        self.assertNotIn("rho_tiles[0", production_source)
        self.assertNotIn("phi_tiles[0", production_source)


if __name__ == "__main__":
    unittest.main()
