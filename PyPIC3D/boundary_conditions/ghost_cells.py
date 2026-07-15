import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


def _reduced_tiled_axes(field_tiles, tile_shape, g):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = field_tiles.shape[:3]
    return (
        int(ntx) == 1 and tile_nx == 1,
        int(nty) == 1 and tile_ny == 1,
        int(ntz) == 1 and tile_nz == 1,
    )


def _is_stacked_tiled_vector_field(field_tiles):
    return hasattr(field_tiles, "ndim") and field_tiles.ndim == 7 and int(field_tiles.shape[0]) == 3


def _stack_tiled_vector_field(field_tiles):
    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles
    return jnp.stack(field_tiles, axis=0)


def _unstack_tiled_vector_field(field_tiles):
    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles[0], field_tiles[1], field_tiles[2]
    return field_tiles


def _restore_tiled_vector_layout(stacked_tiles, original_tiles):
    if _is_stacked_tiled_vector_field(original_tiles):
        return stacked_tiles
    return _unstack_tiled_vector_field(stacked_tiles)


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
    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    reduced_x, reduced_y, reduced_z = _reduced_tiled_axes(field_tiles, tile_shape, g)
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    # x-halos: internal tile boundaries exchange neighboring interiors.  At a
    # conducting global wall, the exterior ghost face is zero, matching the
    # assembled-field boundary convention.
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
    # the sequential ghost-cell convention used by assembled arrays.
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


def update_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tile halos for a tiled vector field.
    """

    stacked_tiles = _stack_tiled_vector_field(field_tiles)
    refreshed = jax.vmap(
        lambda component: update_tiled_ghost_cells(component, world, num_guard_cells, tile_shape),
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

    pml_world = {
        "boundary_conditions": {"x": bc_x, "y": bc_y, "z": bc_z},
        "tile_shape": world["tile_shape"],
    }
    return update_tiled_ghost_cells(field_tiles, pml_world, num_guard_cells, tile_shape)


def update_tiled_vector_ghost_cells_for_pml(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tile halos for a vector field with PML-active exterior walls.
    """

    stacked_tiles = _stack_tiled_vector_field(field_tiles)
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


def apply_tiled_scalar_conducting_bc(field_tiles, world, num_guard_cells=2):
    """
    Zero scalar field values on global conducting faces.
    """

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    g = int(num_guard_cells)
    lower = g
    upper = -g - 1

    field_tiles = jax.lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[0, :, :, lower, :, :].set(0.0).at[-1, :, :, upper, :, :].set(0.0),
        lambda f: f,
        operand=field_tiles,
    )
    field_tiles = jax.lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 0, :, :, lower, :].set(0.0).at[:, -1, :, :, upper, :].set(0.0),
        lambda f: f,
        operand=field_tiles,
    )
    field_tiles = jax.lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 0, :, :, lower].set(0.0).at[:, :, -1, :, :, upper].set(0.0),
        lambda f: f,
        operand=field_tiles,
    )

    return field_tiles


def fold_tiled_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Add tile-ghost deposits into owning interiors, then clear ghosts.

    Internal tile ghosts always belong to the neighboring physical tile
    interior.  At global conducting walls, exterior ghost deposits reflect into
    the adjacent boundary cell with the same sign convention used by the
    legacy global ghost-cell fold.
    """

    g = int(num_guard_cells)
    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
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


def fold_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Fold tile-ghost deposits for a tiled vector field.
    """

    stacked_tiles = _stack_tiled_vector_field(field_tiles)
    folded = jax.vmap(
        lambda component: fold_tiled_ghost_cells(component, world, num_guard_cells, tile_shape),
        in_axes=0,
        out_axes=0,
    )(stacked_tiles)
    return _restore_tiled_vector_layout(folded, field_tiles)
