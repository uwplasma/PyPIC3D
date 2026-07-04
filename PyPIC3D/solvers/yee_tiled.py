import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
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


def _reduced_tiled_axes(field_tiles, tile_shape, g):
    if tile_shape is None:
        tile_shape = tuple(int(width) - 2 * int(g) for width in field_tiles.shape[3:])
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = field_tiles.shape[:3]
    return (
        int(ntx) == 1 and tile_nx == 1,
        int(nty) == 1 and tile_ny == 1,
        int(ntz) == 1 and tile_nz == 1,
    )


def empty_tiled_scalar_field(world, tile_shape, num_guard_cells=2, dtype=None):
    """
    Allocate an empty tile-major scalar field with ``num_guard_cells`` guards.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)
    g = int(num_guard_cells)
    if dtype is None:
        dtype = jnp.float64

    return jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=dtype,
    )


def empty_tiled_vector_field(world, tile_shape, num_guard_cells=2, dtype=None):
    """
    Allocate empty tile-major vector-field components.
    """

    return tuple(empty_tiled_scalar_field(world, tile_shape, num_guard_cells, dtype) for _ in range(3))


def _is_stacked_tiled_vector_field(field_tiles):
    return hasattr(field_tiles, "ndim") and field_tiles.ndim == 7 and int(field_tiles.shape[0]) == 3


def stack_tiled_vector_field(field_tiles):
    """
    Stack tiled vector components onto one leading component axis.

    The public tiled field state still uses the existing ``(Fx, Fy, Fz)``
    tuples.  This packed view gives halo/fold/filter routines one vector-field
    surface, which is the shape later shard-local exchange should operate on.
    """

    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles
    return jnp.stack(field_tiles, axis=0)


def unstack_tiled_vector_field(field_tiles):
    """
    Return a tiled vector field as the existing ``(Fx, Fy, Fz)`` tuple.
    """

    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles[0], field_tiles[1], field_tiles[2]
    return field_tiles


def _restore_tiled_vector_layout(stacked_tiles, original_tiles):
    if _is_stacked_tiled_vector_field(original_tiles):
        return stacked_tiles
    return unstack_tiled_vector_field(stacked_tiles)


def _tile_grid_axis(global_axis_grid, world, tile_shape, tile_counts, axis_index, num_guard_cells):
    tile_width = int(tile_shape[axis_index])
    tile_count = int(tile_counts[axis_index])
    g = int(num_guard_cells)

    if axis_index == 0:
        d = world["dx"]
    elif axis_index == 1:
        d = world["dy"]
    else:
        d = world["dz"]

    offsets = jnp.arange(tile_width + 2 * g, dtype=global_axis_grid.dtype)
    tile_indices = jnp.arange(tile_count, dtype=global_axis_grid.dtype)
    axis_lines = global_axis_grid[0] + (offsets[jnp.newaxis, :] + tile_indices[:, jnp.newaxis] * tile_width - (g - 1)) * d

    axis_shape = [1, 1, 1, tile_width + 2 * g]
    axis_shape[axis_index] = tile_count
    tiled_shape = tuple(tile_counts) + (tile_width + 2 * g,)

    return jnp.broadcast_to(axis_lines.reshape(axis_shape), tiled_shape)


def tile_grid_axes(grid, world, tile_shape, num_guard_cells=2):
    """
    Build tile-local coordinate lines for a center or vertex grid.

    Each returned axis has leading tile indices followed by the local
    ghost-celled coordinate line for that axis.  These arrays carry the same
    coordinate convention previously rebuilt inside tiled interpolation and
    deposition kernels.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    Nx = int(grid[0].shape[0]) - 2
    Ny = int(grid[1].shape[0]) - 2
    Nz = int(grid[2].shape[0]) - 2
    tile_counts = (
        _tile_axis_count(Nx, tile_nx),
        _tile_axis_count(Ny, tile_ny),
        _tile_axis_count(Nz, tile_nz),
    )

    return tuple(
        _tile_grid_axis(grid[axis], world, tile_shape, tile_counts, axis, num_guard_cells)
        for axis in range(3)
    )


def tiled_grid_axes_from_world(world, grid, tiled_grid_key, tile_shape, num_guard_cells=2):
    """
    Read precomputed tile-local grid axes, with a construction fallback.

    Initialization stores the tiled grids in ``world['grids']`` for production
    tiled runs.  The fallback keeps focused tests and older call sites that
    construct small worlds by hand on the same numerical convention.
    """

    grids = world["grids"]
    if tiled_grid_key in grids:
        return grids[tiled_grid_key]

    return tile_grid_axes(grid, world, tile_shape, num_guard_cells)


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
        field_tiles = empty_tiled_scalar_field(world, tile_shape, g, field.dtype)
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = 1 + tx * tile_nx
                    iy = 1 + ty * tile_ny
                    iz = 1 + tz * tile_nz
                    interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                    field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)
        return update_tiled_ghost_cells(field_tiles, world, g, tile_shape)

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


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    """
    Split ``(Fx, Fy, Fz)`` into compact ghost-celled tiles.

    Each scalar component gets leading tile axes ``(ntx, nty, ntz)`` followed by
    the tile-local ghost-celled field shape
    ``(tile_nx + 2, tile_ny + 2, tile_nz + 2)``.
    """

    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


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
                interior = field_tiles[tx, ty, tz, g:-g, g:-g, g:-g]
                ix = 1 + tx * tile_nx
                iy = 1 + ty * tile_ny
                iz = 1 + tz * tile_nz
                field = field.at[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz].set(interior)

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    return update_ghost_cells(field, bc_x, bc_y, bc_z)


def assemble_tiled_vector_field(field_tiles, world, tile_shape, num_guard_cells=2):
    """
    Assemble tiled vector-field components into ordinary ghost-celled arrays.
    """

    return tuple(assemble_tiled_scalar_field(component, world, tile_shape, num_guard_cells) for component in field_tiles)


def _assemble_scalar_tiles(field_tiles, world, tile_shape):
    return assemble_tiled_scalar_field(field_tiles, world, tile_shape)


def update_tiled_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tile halos using the field boundary conditions.

    ``field_tiles`` has shape
    ``(ntx, nty, ntz, tile_nx + 2*g, tile_ny + 2*g, tile_nz + 2*g)``.
    The last three axes are the tile-local ghost-celled field with
    ``g = num_guard_cells`` guard cells on each side.  The first three axes
    identify the tile in the global tiling.
    """

    g = int(num_guard_cells)
    reduced_x, reduced_y, reduced_z = _reduced_tiled_axes(field_tiles, tile_shape, g)
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    # x-halos: internal tile boundaries exchange neighboring interiors.  At a
    # conducting global wall, the exterior ghost face is zero, matching
    # update_ghost_cells on the assembled field.
    if reduced_x:
        lower_x = jnp.broadcast_to(field_tiles[:, :, :, g:g + 1, :, :], field_tiles[:, :, :, :g, :, :].shape)
        upper_x = jnp.broadcast_to(field_tiles[:, :, :, g:g + 1, :, :], field_tiles[:, :, :, -g:, :, :].shape)
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(lower_x)
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(upper_x)
        field_tiles = jax.lax.cond(
            bc_x == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, :g, :, :].set(0.0).at[:, :, :, -g:, :, :].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )
    else:
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(
            jnp.roll(field_tiles[:, :, :, -2 * g:-g, :, :], shift=1, axis=0)
        )
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(
            jnp.roll(field_tiles[:, :, :, g:2 * g, :, :], shift=-1, axis=0)
        )
        field_tiles = jax.lax.cond(
            bc_x == BC_CONDUCTING,
            lambda tiles: tiles.at[0, :, :, :g, :, :].set(0.0).at[-1, :, :, -g:, :, :].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )

    # y-halos use the x-refreshed field so tile-edge/corner guard cells match
    # the sequential ghost-cell convention used by the global arrays.
    if reduced_y:
        lower_y = jnp.broadcast_to(field_tiles[:, :, :, :, g:g + 1, :], field_tiles[:, :, :, :, :g, :].shape)
        upper_y = jnp.broadcast_to(field_tiles[:, :, :, :, g:g + 1, :], field_tiles[:, :, :, :, -g:, :].shape)
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(lower_y)
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(upper_y)
        field_tiles = jax.lax.cond(
            bc_y == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, :, :g, :].set(0.0).at[:, :, :, :, -g:, :].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )
    else:
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(
            jnp.roll(field_tiles[:, :, :, :, -2 * g:-g, :], shift=1, axis=1)
        )
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(
            jnp.roll(field_tiles[:, :, :, :, g:2 * g, :], shift=-1, axis=1)
        )
        field_tiles = jax.lax.cond(
            bc_y == BC_CONDUCTING,
            lambda tiles: tiles.at[:, 0, :, :, :g, :].set(0.0).at[:, -1, :, :, -g:, :].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )

    # z-halos complete the edge and corner values after x/y have been refreshed.
    if reduced_z:
        lower_z = jnp.broadcast_to(field_tiles[:, :, :, :, :, g:g + 1], field_tiles[:, :, :, :, :, :g].shape)
        upper_z = jnp.broadcast_to(field_tiles[:, :, :, :, :, g:g + 1], field_tiles[:, :, :, :, :, -g:].shape)
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(lower_z)
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(upper_z)
        field_tiles = jax.lax.cond(
            bc_z == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, :, :, :g].set(0.0).at[:, :, :, :, :, -g:].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )
    else:
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(
            jnp.roll(field_tiles[:, :, :, :, :, -2 * g:-g], shift=1, axis=2)
        )
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(
            jnp.roll(field_tiles[:, :, :, :, :, g:2 * g], shift=-1, axis=2)
        )
        field_tiles = jax.lax.cond(
            bc_z == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, 0, :, :, :g].set(0.0).at[:, :, -1, :, :, -g:].set(0.0),
            lambda tiles: tiles,
            operand=field_tiles,
        )

    return field_tiles


def update_tiled_ghost_cells_periodic(field_tiles, num_guard_cells=2):
    """
    Refresh tile halos with all-periodic boundary conditions.
    """

    world = {"boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}}
    return update_tiled_ghost_cells(field_tiles, world, num_guard_cells)


def update_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tile halos for a tiled vector field.
    """

    stacked_tiles = stack_tiled_vector_field(field_tiles)
    refreshed = jax.vmap(
        lambda component: update_tiled_ghost_cells(component, world, num_guard_cells, tile_shape),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(refreshed, field_tiles)


def update_tiled_vector_ghost_cells_periodic(field_tiles, num_guard_cells=2):
    """
    Refresh periodic tile halos for each component of a vector field.
    """

    stacked_tiles = stack_tiled_vector_field(field_tiles)
    refreshed = jax.vmap(
        lambda component: update_tiled_ghost_cells_periodic(component, num_guard_cells),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(refreshed, field_tiles)


def update_tiled_ghost_cells_for_pml(field_tiles, world, num_guard_cells=2, tile_shape=None):
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
    return update_tiled_ghost_cells(field_tiles, pml_world, num_guard_cells, tile_shape)


def update_tiled_vector_ghost_cells_for_pml(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tile halos for a vector field with PML-active exterior walls.
    """

    stacked_tiles = stack_tiled_vector_field(field_tiles)
    refreshed = jax.vmap(
        lambda component: update_tiled_ghost_cells_for_pml(component, world, num_guard_cells, tile_shape),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(refreshed, field_tiles)


def apply_tiled_conducting_bc(E_tiles, world, num_guard_cells=2):
    """
    Zero tangential electric-field components on global conducting faces.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    Ex, Ey, Ez = E_tiles
    g = int(num_guard_cells)
    lower = g
    upper = -g - 1

    Ey = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[0, :, :, lower, :, :].set(0.0).at[-1, :, :, upper, :, :].set(0.0),
        lambda f: f,
        operand=Ey,
    )
    Ez = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[0, :, :, lower, :, :].set(0.0).at[-1, :, :, upper, :, :].set(0.0),
        lambda f: f,
        operand=Ez,
    )

    Ex = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 0, :, :, lower, :].set(0.0).at[:, -1, :, :, upper, :].set(0.0),
        lambda f: f,
        operand=Ex,
    )
    Ez = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 0, :, :, lower, :].set(0.0).at[:, -1, :, :, upper, :].set(0.0),
        lambda f: f,
        operand=Ez,
    )

    Ex = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 0, :, :, lower].set(0.0).at[:, :, -1, :, :, upper].set(0.0),
        lambda f: f,
        operand=Ex,
    )
    Ey = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 0, :, :, lower].set(0.0).at[:, :, -1, :, :, upper].set(0.0),
        lambda f: f,
        operand=Ey,
    )

    return Ex, Ey, Ez


def fold_tiled_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Add tile-ghost deposits into owning interiors, then clear ghosts.

    This is the current-deposition analogue of ``fold_ghost_cells`` for a
    tiled layout.  Internal tile ghosts always belong to the neighboring
    physical tile interior.  At global conducting walls, exterior ghost
    deposits reflect into the adjacent boundary cell with the sign convention
    used by the global ghost-cell fold.
    """

    g = int(num_guard_cells)
    reduced_x, reduced_y, reduced_z = _reduced_tiled_axes(field_tiles, tile_shape, g)
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    if reduced_x:
        x_ghost_sum = (
            jnp.sum(field_tiles[:, :, :, :g, :, :], axis=3, keepdims=True)
            + jnp.sum(field_tiles[:, :, :, -g:, :, :], axis=3, keepdims=True)
        )
        field_tiles = jax.lax.cond(
            bc_x == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, g:g + 1, :, :].add(-x_ghost_sum),
            lambda tiles: tiles.at[:, :, :, g:g + 1, :, :].add(x_ghost_sum),
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(0.0)
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(0.0)
    else:
        x_lower_ghost = field_tiles[:, :, :, :g, :, :]
        x_upper_ghost = field_tiles[:, :, :, -g:, :, :]
        field_tiles = field_tiles.at[:, :, :, -2 * g:-g, :, :].add(
            jnp.roll(x_lower_ghost, shift=-1, axis=0)
        )
        field_tiles = field_tiles.at[:, :, :, g:2 * g, :, :].add(
            jnp.roll(x_upper_ghost, shift=1, axis=0)
        )
        field_tiles = jax.lax.cond(
            bc_x == BC_CONDUCTING,
            lambda tiles: tiles
            .at[-1, :, :, -2 * g:-g, :, :].add(-x_lower_ghost[0, :, :, :, :])
            .at[0, :, :, g:2 * g, :, :].add(-x_upper_ghost[-1, :, :, :, :])
            .at[0, :, :, g:2 * g, :, :].add(-x_lower_ghost[0, :, :, :, :])
            .at[-1, :, :, -2 * g:-g, :, :].add(-x_upper_ghost[-1, :, :, :, :]),
            lambda tiles: tiles,
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(0.0)
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(0.0)

    if reduced_y:
        y_ghost_sum = (
            jnp.sum(field_tiles[:, :, :, :, :g, :], axis=4, keepdims=True)
            + jnp.sum(field_tiles[:, :, :, :, -g:, :], axis=4, keepdims=True)
        )
        field_tiles = jax.lax.cond(
            bc_y == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, :, g:g + 1, :].add(-y_ghost_sum),
            lambda tiles: tiles.at[:, :, :, :, g:g + 1, :].add(y_ghost_sum),
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(0.0)
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(0.0)
    else:
        y_lower_ghost = field_tiles[:, :, :, :, :g, :]
        y_upper_ghost = field_tiles[:, :, :, :, -g:, :]
        field_tiles = field_tiles.at[:, :, :, :, -2 * g:-g, :].add(
            jnp.roll(y_lower_ghost, shift=-1, axis=1)
        )
        field_tiles = field_tiles.at[:, :, :, :, g:2 * g, :].add(
            jnp.roll(y_upper_ghost, shift=1, axis=1)
        )
        field_tiles = jax.lax.cond(
            bc_y == BC_CONDUCTING,
            lambda tiles: tiles
            .at[:, -1, :, :, -2 * g:-g, :].add(-y_lower_ghost[:, 0, :, :, :])
            .at[:, 0, :, :, g:2 * g, :].add(-y_upper_ghost[:, -1, :, :, :])
            .at[:, 0, :, :, g:2 * g, :].add(-y_lower_ghost[:, 0, :, :, :])
            .at[:, -1, :, :, -2 * g:-g, :].add(-y_upper_ghost[:, -1, :, :, :]),
            lambda tiles: tiles,
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(0.0)
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(0.0)

    if reduced_z:
        z_ghost_sum = (
            jnp.sum(field_tiles[:, :, :, :, :, :g], axis=5, keepdims=True)
            + jnp.sum(field_tiles[:, :, :, :, :, -g:], axis=5, keepdims=True)
        )
        field_tiles = jax.lax.cond(
            bc_z == BC_CONDUCTING,
            lambda tiles: tiles.at[:, :, :, :, :, g:g + 1].add(-z_ghost_sum),
            lambda tiles: tiles.at[:, :, :, :, :, g:g + 1].add(z_ghost_sum),
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(0.0)
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(0.0)
    else:
        z_lower_ghost = field_tiles[:, :, :, :, :, :g]
        z_upper_ghost = field_tiles[:, :, :, :, :, -g:]
        field_tiles = field_tiles.at[:, :, :, :, :, -2 * g:-g].add(
            jnp.roll(z_lower_ghost, shift=-1, axis=2)
        )
        field_tiles = field_tiles.at[:, :, :, :, :, g:2 * g].add(
            jnp.roll(z_upper_ghost, shift=1, axis=2)
        )
        field_tiles = jax.lax.cond(
            bc_z == BC_CONDUCTING,
            lambda tiles: tiles
            .at[:, :, -1, :, :, -2 * g:-g].add(-z_lower_ghost[:, :, 0, :, :])
            .at[:, :, 0, :, :, g:2 * g].add(-z_upper_ghost[:, :, -1, :, :])
            .at[:, :, 0, :, :, g:2 * g].add(-z_lower_ghost[:, :, 0, :, :])
            .at[:, :, -1, :, :, -2 * g:-g].add(-z_upper_ghost[:, :, -1, :, :]),
            lambda tiles: tiles,
            operand=field_tiles,
        )
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(0.0)
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(0.0)

    return field_tiles


def fold_tiled_ghost_cells_periodic(field_tiles, num_guard_cells=2):
    """
    Fold all-periodic tile-ghost deposits for one scalar component.
    """

    world = {"boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}}
    return fold_tiled_ghost_cells(field_tiles, world, num_guard_cells)


def fold_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Fold tile-ghost deposits for a tiled vector field.
    """

    stacked_tiles = stack_tiled_vector_field(field_tiles)
    folded = jax.vmap(
        lambda component: fold_tiled_ghost_cells(component, world, num_guard_cells, tile_shape),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(folded, field_tiles)


def fold_tiled_vector_ghost_cells_periodic(field_tiles, num_guard_cells=2):
    """
    Fold all-periodic tile-ghost deposits for each vector component.
    """

    stacked_tiles = stack_tiled_vector_field(field_tiles)
    folded = jax.vmap(
        lambda component: fold_tiled_ghost_cells_periodic(component, num_guard_cells),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(folded, field_tiles)


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
        Bx, By, Bz = update_tiled_vector_ghost_cells(B_tiles, world, g, tile_shape)
    else:
        Bx, By, Bz = update_tiled_vector_ghost_cells_for_pml(B_tiles, world, g, tile_shape)
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
        Ex, Ey, Ez = update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g, tile_shape)
    else:
        Ex, Ey, Ez = update_tiled_vector_ghost_cells_for_pml((Ex, Ey, Ez), world, g, tile_shape)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Ex, Ey, Ez = digital_filter_vector((Ex, Ey, Ez), constants.get("alpha", 1.0), num_guard_cells=g)

    Ex, Ey, Ez = apply_tiled_conducting_bc((Ex, Ey, Ez), world, num_guard_cells=g)

    if pml_state is None:
        return update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g, tile_shape), None

    return update_tiled_vector_ghost_cells_for_pml((Ex, Ey, Ez), world, g, tile_shape), pml_state


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
        Ex, Ey, Ez = update_tiled_vector_ghost_cells(E_tiles, world, g, tile_shape)
    else:
        Ex, Ey, Ez = update_tiled_vector_ghost_cells_for_pml(E_tiles, world, g, tile_shape)
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
        Bx, By, Bz = update_tiled_vector_ghost_cells((Bx, By, Bz), world, g, tile_shape)
    else:
        Bx, By, Bz = update_tiled_vector_ghost_cells_for_pml((Bx, By, Bz), world, g, tile_shape)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Bx, By, Bz = digital_filter_vector((Bx, By, Bz), constants.get("alpha", 1.0), num_guard_cells=g)

    if pml_state is None:
        return update_tiled_vector_ghost_cells((Bx, By, Bz), world, g, tile_shape), None

    return update_tiled_vector_ghost_cells_for_pml((Bx, By, Bz), world, g, tile_shape), pml_state
