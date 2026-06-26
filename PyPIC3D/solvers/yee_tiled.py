import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape):
    """
    Split a ghost-celled field into compact tiles using the shared tile shape.
    """

    del world

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

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


def _tile_scalar_field(field, tile_shape):
    return tile_scalar_field(field, None, tile_shape)


def tile_vector_field(field, world, tile_shape):
    """
    Split ``(Fx, Fy, Fz)`` into compact ghost-celled tiles.

    Each scalar component gets leading tile axes ``(ntx, nty, ntz)`` followed by
    the tile-local ghost-celled field shape
    ``(tile_nx + 2, tile_ny + 2, tile_nz + 2)``.
    """

    return tuple(tile_scalar_field(component, world, tile_shape) for component in field)


def assemble_tiled_scalar_field(field_tiles, world, tile_shape):
    """
    Assemble compact field tiles back into one global ghost-celled field.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = field_tiles.shape[:3]
    Nx = int(ntx) * tile_nx
    Ny = int(nty) * tile_ny
    Nz = int(ntz) * tile_nz

    field = jnp.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=field_tiles.dtype)

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                interior = field_tiles[tx, ty, tz, 1:-1, 1:-1, 1:-1]
                ix = 1 + tx * tile_nx
                iy = 1 + ty * tile_ny
                iz = 1 + tz * tile_nz
                field = field.at[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz].set(interior)

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    return update_ghost_cells(field, bc_x, bc_y, bc_z)


def assemble_tiled_vector_field(field_tiles, world, tile_shape):
    """
    Assemble tiled vector-field components into ordinary ghost-celled arrays.
    """

    return tuple(assemble_tiled_scalar_field(component, world, tile_shape) for component in field_tiles)


def _assemble_scalar_tiles(field_tiles, world, tile_shape):
    return assemble_tiled_scalar_field(field_tiles, world, tile_shape)


def update_tiled_ghost_cells(field_tiles, world):
    """
    Refresh one-cell tile halos using the field boundary conditions.

    ``field_tiles`` has shape
    ``(ntx, nty, ntz, tile_nx + 2, tile_ny + 2, tile_nz + 2)``.  The last
    three axes are the tile-local ghost-celled field.  The first three axes
    identify the tile in the global tiling.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    # x-halos: internal tile boundaries exchange neighboring interiors.  At a
    # conducting global wall, the exterior ghost face is zero, matching
    # update_ghost_cells on the assembled field.
    field_tiles = field_tiles.at[:, :, :, 0, :, :].set(
        jnp.roll(field_tiles[:, :, :, -2, :, :], shift=1, axis=0)
    )
    field_tiles = field_tiles.at[:, :, :, -1, :, :].set(
        jnp.roll(field_tiles[:, :, :, 1, :, :], shift=-1, axis=0)
    )
    field_tiles = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda tiles: tiles.at[0, :, :, 0, :, :].set(0.0).at[-1, :, :, -1, :, :].set(0.0),
        lambda tiles: tiles,
        operand=field_tiles,
    )

    # y-halos use the x-refreshed field so tile-edge/corner guard cells match
    # the sequential ghost-cell convention used by the global arrays.
    field_tiles = field_tiles.at[:, :, :, :, 0, :].set(
        jnp.roll(field_tiles[:, :, :, :, -2, :], shift=1, axis=1)
    )
    field_tiles = field_tiles.at[:, :, :, :, -1, :].set(
        jnp.roll(field_tiles[:, :, :, :, 1, :], shift=-1, axis=1)
    )
    field_tiles = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda tiles: tiles.at[:, 0, :, :, 0, :].set(0.0).at[:, -1, :, :, -1, :].set(0.0),
        lambda tiles: tiles,
        operand=field_tiles,
    )

    # z-halos complete the edge and corner values after x/y have been refreshed.
    field_tiles = field_tiles.at[:, :, :, :, :, 0].set(
        jnp.roll(field_tiles[:, :, :, :, :, -2], shift=1, axis=2)
    )
    field_tiles = field_tiles.at[:, :, :, :, :, -1].set(
        jnp.roll(field_tiles[:, :, :, :, :, 1], shift=-1, axis=2)
    )
    field_tiles = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda tiles: tiles.at[:, :, 0, :, :, 0].set(0.0).at[:, :, -1, :, :, -1].set(0.0),
        lambda tiles: tiles,
        operand=field_tiles,
    )

    return field_tiles


def update_tiled_ghost_cells_periodic(field_tiles):
    """
    Refresh one-cell tile halos with all-periodic boundary conditions.
    """

    world = {"boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}}
    return update_tiled_ghost_cells(field_tiles, world)


def update_tiled_vector_ghost_cells(field_tiles, world):
    """
    Refresh tile halos for each component of a vector field.
    """

    return tuple(update_tiled_ghost_cells(component, world) for component in field_tiles)


def update_tiled_vector_ghost_cells_periodic(field_tiles):
    """
    Refresh periodic tile halos for each component of a vector field.
    """

    return tuple(update_tiled_ghost_cells_periodic(component) for component in field_tiles)


def update_tiled_ghost_cells_for_pml(field_tiles, world):
    """
    Refresh tile halos without wrapping across PML-active global walls.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    _, pml_x, pml_y, pml_z, _ = world["pml"]

    bc_x = jnp.where((pml_x) & (bc_x == BC_PERIODIC), BC_CONDUCTING, bc_x)
    bc_y = jnp.where((pml_y) & (bc_y == BC_PERIODIC), BC_CONDUCTING, bc_y)
    bc_z = jnp.where((pml_z) & (bc_z == BC_PERIODIC), BC_CONDUCTING, bc_z)

    pml_world = {"boundary_conditions": {"x": bc_x, "y": bc_y, "z": bc_z}}
    return update_tiled_ghost_cells(field_tiles, pml_world)


def update_tiled_vector_ghost_cells_for_pml(field_tiles, world):
    """
    Refresh tile halos for a vector field with PML-active exterior walls.
    """

    return tuple(update_tiled_ghost_cells_for_pml(component, world) for component in field_tiles)


def apply_tiled_conducting_bc(E_tiles, world):
    """
    Zero tangential electric-field components on global conducting faces.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    Ex, Ey, Ez = E_tiles

    Ey = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[0, :, :, 1, :, :].set(0.0).at[-1, :, :, -2, :, :].set(0.0),
        lambda f: f,
        operand=Ey,
    )
    Ez = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[0, :, :, 1, :, :].set(0.0).at[-1, :, :, -2, :, :].set(0.0),
        lambda f: f,
        operand=Ez,
    )

    Ex = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 0, :, :, 1, :].set(0.0).at[:, -1, :, :, -2, :].set(0.0),
        lambda f: f,
        operand=Ex,
    )
    Ez = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 0, :, :, 1, :].set(0.0).at[:, -1, :, :, -2, :].set(0.0),
        lambda f: f,
        operand=Ez,
    )

    Ex = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 0, :, :, 1].set(0.0).at[:, :, -1, :, :, -2].set(0.0),
        lambda f: f,
        operand=Ex,
    )
    Ey = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 0, :, :, 1].set(0.0).at[:, :, -1, :, :, -2].set(0.0),
        lambda f: f,
        operand=Ey,
    )

    return Ex, Ey, Ez


def fold_tiled_ghost_cells(field_tiles, world):
    """
    Add tile-ghost deposits into owning interiors, then clear ghosts.

    This is the current-deposition analogue of ``fold_ghost_cells`` for a
    tiled layout.  Internal tile ghosts always belong to the neighboring
    physical tile interior.  At global conducting walls, exterior ghost
    deposits reflect into the adjacent boundary cell with the sign convention
    used by the global ghost-cell fold.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    x_lower_ghost = field_tiles[:, :, :, 0, :, :]
    x_upper_ghost = field_tiles[:, :, :, -1, :, :]
    field_tiles = field_tiles.at[:, :, :, -2, :, :].add(
        jnp.roll(x_lower_ghost, shift=-1, axis=0)
    )
    field_tiles = field_tiles.at[:, :, :, 1, :, :].add(
        jnp.roll(x_upper_ghost, shift=1, axis=0)
    )
    field_tiles = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda tiles: tiles
        .at[-1, :, :, -2, :, :].add(-x_lower_ghost[0, :, :, :, :])
        .at[0, :, :, 1, :, :].add(-x_upper_ghost[-1, :, :, :, :])
        .at[0, :, :, 1, :, :].add(-x_lower_ghost[0, :, :, :, :])
        .at[-1, :, :, -2, :, :].add(-x_upper_ghost[-1, :, :, :, :]),
        lambda tiles: tiles,
        operand=field_tiles,
    )
    field_tiles = field_tiles.at[:, :, :, 0, :, :].set(0.0)
    field_tiles = field_tiles.at[:, :, :, -1, :, :].set(0.0)

    y_lower_ghost = field_tiles[:, :, :, :, 0, :]
    y_upper_ghost = field_tiles[:, :, :, :, -1, :]
    field_tiles = field_tiles.at[:, :, :, :, -2, :].add(
        jnp.roll(y_lower_ghost, shift=-1, axis=1)
    )
    field_tiles = field_tiles.at[:, :, :, :, 1, :].add(
        jnp.roll(y_upper_ghost, shift=1, axis=1)
    )
    field_tiles = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda tiles: tiles
        .at[:, -1, :, :, -2, :].add(-y_lower_ghost[:, 0, :, :, :])
        .at[:, 0, :, :, 1, :].add(-y_upper_ghost[:, -1, :, :, :])
        .at[:, 0, :, :, 1, :].add(-y_lower_ghost[:, 0, :, :, :])
        .at[:, -1, :, :, -2, :].add(-y_upper_ghost[:, -1, :, :, :]),
        lambda tiles: tiles,
        operand=field_tiles,
    )
    field_tiles = field_tiles.at[:, :, :, :, 0, :].set(0.0)
    field_tiles = field_tiles.at[:, :, :, :, -1, :].set(0.0)

    z_lower_ghost = field_tiles[:, :, :, :, :, 0]
    z_upper_ghost = field_tiles[:, :, :, :, :, -1]
    field_tiles = field_tiles.at[:, :, :, :, :, -2].add(
        jnp.roll(z_lower_ghost, shift=-1, axis=2)
    )
    field_tiles = field_tiles.at[:, :, :, :, :, 1].add(
        jnp.roll(z_upper_ghost, shift=1, axis=2)
    )
    field_tiles = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda tiles: tiles
        .at[:, :, -1, :, :, -2].add(-z_lower_ghost[:, :, 0, :, :])
        .at[:, :, 0, :, :, 1].add(-z_upper_ghost[:, :, -1, :, :])
        .at[:, :, 0, :, :, 1].add(-z_lower_ghost[:, :, 0, :, :])
        .at[:, :, -1, :, :, -2].add(-z_upper_ghost[:, :, -1, :, :]),
        lambda tiles: tiles,
        operand=field_tiles,
    )
    field_tiles = field_tiles.at[:, :, :, :, :, 0].set(0.0)
    field_tiles = field_tiles.at[:, :, :, :, :, -1].set(0.0)

    return field_tiles


def fold_tiled_ghost_cells_periodic(field_tiles):
    """
    Fold all-periodic tile-ghost deposits for one scalar component.
    """

    world = {"boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}}
    return fold_tiled_ghost_cells(field_tiles, world)


def fold_tiled_vector_ghost_cells(field_tiles, world):
    """
    Fold tile-ghost deposits for each vector component.
    """

    return tuple(fold_tiled_ghost_cells(component, world) for component in field_tiles)


def fold_tiled_vector_ghost_cells_periodic(field_tiles):
    """
    Fold all-periodic tile-ghost deposits for each vector component.
    """

    return tuple(fold_tiled_ghost_cells_periodic(component) for component in field_tiles)


def update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after B halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    del curl_func, tile_shape

    Ex, Ey, Ez = E_tiles
    if pml_state is None:
        Bx, By, Bz = update_tiled_vector_ghost_cells(B_tiles, world)
    else:
        Bx, By, Bz = update_tiled_vector_ghost_cells_for_pml(B_tiles, world)
    Jx, Jy, Jz = J_tiles

    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    C = constants["C"]
    eps = constants["eps"]

    # Forward differences use each tile's + side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dBz_dy = (Bz[:, :, :, 1:-1, 2:, 1:-1] - Bz[:, :, :, 1:-1, 1:-1, 1:-1]) / dy
    dBy_dz = (By[:, :, :, 1:-1, 1:-1, 2:] - By[:, :, :, 1:-1, 1:-1, 1:-1]) / dz
    dBx_dz = (Bx[:, :, :, 1:-1, 1:-1, 2:] - Bx[:, :, :, 1:-1, 1:-1, 1:-1]) / dz
    dBx_dy = (Bx[:, :, :, 1:-1, 2:, 1:-1] - Bx[:, :, :, 1:-1, 1:-1, 1:-1]) / dy
    dBz_dx = (Bz[:, :, :, 2:, 1:-1, 1:-1] - Bz[:, :, :, 1:-1, 1:-1, 1:-1]) / dx
    dBy_dx = (By[:, :, :, 2:, 1:-1, 1:-1] - By[:, :, :, 1:-1, 1:-1, 1:-1]) / dx

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

    Ex = Ex.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        Ex[:, :, :, 1:-1, 1:-1, 1:-1]
        + (C**2 * curl_x - Jx[:, :, :, 1:-1, 1:-1, 1:-1] / eps) * dt
    )
    Ey = Ey.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        Ey[:, :, :, 1:-1, 1:-1, 1:-1]
        + (C**2 * curl_y - Jy[:, :, :, 1:-1, 1:-1, 1:-1] / eps) * dt
    )
    Ez = Ez.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        Ez[:, :, :, 1:-1, 1:-1, 1:-1]
        + (C**2 * curl_z - Jz[:, :, :, 1:-1, 1:-1, 1:-1] / eps) * dt
    )

    Ex, Ey, Ez = apply_tiled_conducting_bc((Ex, Ey, Ez), world)

    if pml_state is None:
        return update_tiled_vector_ghost_cells((Ex, Ey, Ez), world)

    return update_tiled_vector_ghost_cells_for_pml((Ex, Ey, Ez), world), pml_state


def update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape, pml_state=None):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    del constants, curl_func, tile_shape

    if pml_state is None:
        Ex, Ey, Ez = update_tiled_vector_ghost_cells(E_tiles, world)
    else:
        Ex, Ey, Ez = update_tiled_vector_ghost_cells_for_pml(E_tiles, world)
    Bx, By, Bz = B_tiles

    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]

    # Backward differences use each tile's - side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dEz_dy = (Ez[:, :, :, 1:-1, 1:-1, 1:-1] - Ez[:, :, :, 1:-1, :-2, 1:-1]) / dy
    dEy_dz = (Ey[:, :, :, 1:-1, 1:-1, 1:-1] - Ey[:, :, :, 1:-1, 1:-1, :-2]) / dz
    dEx_dz = (Ex[:, :, :, 1:-1, 1:-1, 1:-1] - Ex[:, :, :, 1:-1, 1:-1, :-2]) / dz
    dEx_dy = (Ex[:, :, :, 1:-1, 1:-1, 1:-1] - Ex[:, :, :, 1:-1, :-2, 1:-1]) / dy
    dEz_dx = (Ez[:, :, :, 1:-1, 1:-1, 1:-1] - Ez[:, :, :, :-2, 1:-1, 1:-1]) / dx
    dEy_dx = (Ey[:, :, :, 1:-1, 1:-1, 1:-1] - Ey[:, :, :, :-2, 1:-1, 1:-1]) / dx

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

    Bx = Bx.at[:, :, :, 1:-1, 1:-1, 1:-1].set(Bx[:, :, :, 1:-1, 1:-1, 1:-1] - dt * curl_x)
    By = By.at[:, :, :, 1:-1, 1:-1, 1:-1].set(By[:, :, :, 1:-1, 1:-1, 1:-1] - dt * curl_y)
    Bz = Bz.at[:, :, :, 1:-1, 1:-1, 1:-1].set(Bz[:, :, :, 1:-1, 1:-1, 1:-1] - dt * curl_z)

    if pml_state is None:
        return update_tiled_vector_ghost_cells((Bx, By, Bz), world)

    return update_tiled_vector_ghost_cells_for_pml((Bx, By, Bz), world), pml_state
