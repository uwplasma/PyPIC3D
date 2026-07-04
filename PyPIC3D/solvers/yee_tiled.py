import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.utilities.filters import digital_filter_vector


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def _active_slice(g):
    return slice(g, -g)


def _forward_slice(g):
    return slice(g + 1, None if g == 1 else -g + 1)


def _backward_slice(g):
    return slice(g - 1, -g - 1)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    """
    Split a ghost-celled field into compact tiles using the shared tile shape.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    if g != 1:
        field_tiles = jnp.zeros(
            (
                ntx,
                nty,
                ntz,
                tile_nx + 2 * g,
                tile_ny + 2 * g,
                tile_nz + 2 * g,
            ),
            dtype=field.dtype,
        )
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = 1 + tx * tile_nx
                    iy = 1 + ty * tile_ny
                    iz = 1 + tz * tile_nz
                    interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                    field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)
        return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(field, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def assemble_tiled_scalar_field(field_tiles, world, tile_shape, num_guard_cells=2):
    """
    Assemble compact field tiles back into one global ghost-celled field.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    ntx, nty, ntz = field_tiles.shape[:3]
    Nx = int(ntx) * tile_nx
    Ny = int(nty) * tile_ny
    Nz = int(ntz) * tile_nz

    field = jnp.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=field_tiles.dtype)

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                tile_with_one_guard = field_tiles[
                    tx,
                    ty,
                    tz,
                    g - 1:g + tile_nx + 1,
                    g - 1:g + tile_ny + 1,
                    g - 1:g + tile_nz + 1,
                ]
                ix = tx * tile_nx
                iy = ty * tile_ny
                iz = tz * tile_nz
                field = field.at[ix:ix + tile_nx + 2, iy:iy + tile_ny + 2, iz:iz + tile_nz + 2].set(tile_with_one_guard)

    return field


def assemble_tiled_vector_field(field_tiles, world, tile_shape, num_guard_cells=2):
    """
    Assemble tiled vector-field components into ordinary ghost-celled arrays.
    """

    return tuple(assemble_tiled_scalar_field(component, world, tile_shape, num_guard_cells) for component in field_tiles)


def update_E(E_tiles, B_tiles, J_tiles, world, constants, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after B halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Ex, Ey, Ez = E_tiles
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    tile_nx, tile_ny, tile_nz = tile_shape
    g = int(world["guard_cells"])
    g = int(g)
    active = _active_slice(g)
    forward = _forward_slice(g)
    if pml_state is None:
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells(B_tiles, world, g, tile_shape)
    else:
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells_for_pml(B_tiles, world, g, tile_shape)
    Jx, Jy, Jz = J_tiles
    current = _active_slice(g)

    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    C = constants["C"]
    eps = constants["eps"]

    # Forward differences use each tile's + side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dBz_dy = (Bz[:, :, :, active, forward, active] - Bz[:, :, :, active, active, active]) / dy
    dBy_dz = (By[:, :, :, active, active, forward] - By[:, :, :, active, active, active]) / dz
    dBx_dz = (Bx[:, :, :, active, active, forward] - Bx[:, :, :, active, active, active]) / dz
    dBx_dy = (Bx[:, :, :, active, forward, active] - Bx[:, :, :, active, active, active]) / dy
    dBz_dx = (Bz[:, :, :, forward, active, active] - Bz[:, :, :, active, active, active]) / dx
    dBy_dx = (By[:, :, :, forward, active, active] - By[:, :, :, active, active, active]) / dx

    if pml_state is None:
        curl_x = dBz_dy - dBy_dz
        curl_y = dBx_dz - dBz_dx
        curl_z = dBy_dx - dBx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_e_curl(
            (dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy),
            world,
            pml_state,
        )

    Ex = Ex.at[:, :, :, active, active, active].set(
        Ex[:, :, :, active, active, active]
        + (C**2 * curl_x - Jx[:, :, :, current, current, current] / eps) * dt
    )
    Ey = Ey.at[:, :, :, active, active, active].set(
        Ey[:, :, :, active, active, active]
        + (C**2 * curl_y - Jy[:, :, :, current, current, current] / eps) * dt
    )
    Ez = Ez.at[:, :, :, active, active, active].set(
        Ez[:, :, :, active, active, active]
        + (C**2 * curl_z - Jz[:, :, :, current, current, current] / eps) * dt
    )

    if pml_state is None:
        Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g, tile_shape)
    else:
        Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells_for_pml((Ex, Ey, Ez), world, g, tile_shape)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Ex, Ey, Ez = digital_filter_vector((Ex, Ey, Ez), constants.get("alpha", 1.0), num_guard_cells=g)

    Ex, Ey, Ez = ghost_cells.apply_tiled_conducting_bc((Ex, Ey, Ez), world, num_guard_cells=g)

    if pml_state is None:
        return ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g, tile_shape), None

    return ghost_cells.update_tiled_vector_ghost_cells_for_pml((Ex, Ey, Ez), world, g, tile_shape), pml_state


def update_B(E_tiles, B_tiles, world, constants, pml_state=None):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Bx, By, Bz = B_tiles
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    tile_nx, tile_ny, tile_nz = tile_shape
    g = int(world["guard_cells"])
    g = int(g)
    active = _active_slice(g)
    backward = _backward_slice(g)
    if pml_state is None:
        Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, world, g, tile_shape)
    else:
        Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells_for_pml(E_tiles, world, g, tile_shape)
    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]

    # Backward differences use each tile's - side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dEz_dy = (Ez[:, :, :, active, active, active] - Ez[:, :, :, active, backward, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, active] - Ey[:, :, :, active, active, backward]) / dz
    dEx_dz = (Ex[:, :, :, active, active, active] - Ex[:, :, :, active, active, backward]) / dz
    dEx_dy = (Ex[:, :, :, active, active, active] - Ex[:, :, :, active, backward, active]) / dy
    dEz_dx = (Ez[:, :, :, active, active, active] - Ez[:, :, :, backward, active, active]) / dx
    dEy_dx = (Ey[:, :, :, active, active, active] - Ey[:, :, :, backward, active, active]) / dx

    if pml_state is None:
        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_b_curl(
            (dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy),
            world,
            pml_state,
        )

    Bx = Bx.at[:, :, :, active, active, active].set(Bx[:, :, :, active, active, active] - dt * curl_x)
    By = By.at[:, :, :, active, active, active].set(By[:, :, :, active, active, active] - dt * curl_y)
    Bz = Bz.at[:, :, :, active, active, active].set(Bz[:, :, :, active, active, active] - dt * curl_z)

    if pml_state is None:
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), world, g, tile_shape)
    else:
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells_for_pml((Bx, By, Bz), world, g, tile_shape)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Bx, By, Bz = digital_filter_vector((Bx, By, Bz), constants.get("alpha", 1.0), num_guard_cells=g)

    if pml_state is None:
        return ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), world, g, tile_shape), None

    return ghost_cells.update_tiled_vector_ghost_cells_for_pml((Bx, By, Bz), world, g, tile_shape), pml_state
