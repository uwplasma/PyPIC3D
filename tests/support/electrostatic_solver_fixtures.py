import unittest

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _apply_tiled_phi_constant_boundaries,
    _poisson_residual,
    _tiled_laplacian,
    solve_poisson_with_tiled_local_schwarz,
)
from tests.kernel_fixtures import kernel_parameters


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

    tile_grid_shape = (1, 1, 1)
    tile_shape = (cells_per_axis,) * 3
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


class TiledLocalSchwarzFixtures:
    def _require_devices(self, count):
        if jax.device_count() < count:
            raise RuntimeError(f"Need {count} JAX devices, got {jax.device_count()}")
