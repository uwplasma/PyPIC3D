import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None, None)
VECTOR_TILE_SPEC = P(None, "tile_x", "tile_y", "tile_z", None, None, None)


def _as_python_int(value):
    return int(np.asarray(value))


def _boundary_tuple(boundary_conditions):
    return (
        _as_python_int(boundary_conditions["x"]),
        _as_python_int(boundary_conditions["y"]),
        _as_python_int(boundary_conditions["z"]),
    )


def _effective_pml_boundary_tuple(world):
    bc_x, bc_y, bc_z = _boundary_tuple(world["boundary_conditions"])
    _, pml_x, pml_y, pml_z, _ = world["pml"]

    if bool(pml_x) and bc_x == BC_PERIODIC:
        bc_x = BC_CONDUCTING
    if bool(pml_y) and bc_y == BC_PERIODIC:
        bc_y = BC_CONDUCTING
    if bool(pml_z) and bc_z == BC_PERIODIC:
        bc_z = BC_CONDUCTING

    return bc_x, bc_y, bc_z


def _reduced_axes_from_tile_shape(tile_shape, mesh_shape):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = [int(width) for width in mesh_shape]
    return (
        tile_nx == 1 and ntx == 1,
        tile_ny == 1 and nty == 1,
        tile_nz == 1 and ntz == 1,
    )


def _is_stacked_tiled_vector_field(field_tiles):
    return hasattr(field_tiles, "ndim") and field_tiles.ndim == 7 and int(field_tiles.shape[0]) == 3


def _stack_tiled_vector_field(field_tiles):
    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles
    return jnp.stack(field_tiles, axis=0)


def _unstack_tiled_vector_field(field_tiles):
    return field_tiles[0], field_tiles[1], field_tiles[2]


def _restore_tiled_vector_layout(stacked_tiles, original_tiles):
    if _is_stacked_tiled_vector_field(original_tiles):
        return stacked_tiles
    return _unstack_tiled_vector_field(stacked_tiles)


def _send_positive_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == BC_PERIODIC:
        return tuple((i, (i + 1) % axis_size) for i in range(axis_size))
    return tuple((i, i + 1) for i in range(axis_size - 1))


def _send_negative_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == BC_PERIODIC:
        return tuple((i, (i - 1) % axis_size) for i in range(axis_size))
    return tuple((i, i - 1) for i in range(1, axis_size))


def _axis_permutations(mesh_shape, boundary_conditions):
    send_positive = tuple(
        _send_positive_permutation(axis_size, bc)
        for axis_size, bc in zip(mesh_shape, boundary_conditions)
    )
    send_negative = tuple(
        _send_negative_permutation(axis_size, bc)
        for axis_size, bc in zip(mesh_shape, boundary_conditions)
    )
    return send_positive, send_negative


def _default_mesh_for_tile_shape(tile_grid_shape):
    tile_grid_shape = tuple(int(width) for width in tile_grid_shape)
    n_devices = math.prod(tile_grid_shape)
    devices = jax.devices()
    if len(devices) < n_devices:
        raise ValueError(
            "Tiled field communication requires one logical tile per device: "
            f"tile topology {tile_grid_shape} needs {n_devices} devices, "
            f"but JAX exposes {len(devices)}."
        )
    return Mesh(np.asarray(devices[:n_devices]).reshape(tile_grid_shape), MESH_AXES)


def make_field_mesh(tile_grid_shape):
    """
    Build the JAX device mesh for one logical field tile per device.
    """

    return _default_mesh_for_tile_shape(tile_grid_shape)


def _mesh_from_world_or_field(world, field_tiles, vector=False):
    if world is not None and "field_mesh" in world:
        return world["field_mesh"]
    if world is not None and "mesh" in world:
        return world["mesh"]

    tile_grid_shape = tuple(field_tiles.shape[1:4] if vector else field_tiles.shape[:3])
    return _default_mesh_for_tile_shape(tile_grid_shape)


def _validate_scalar_tile_topology(field_tiles, mesh):
    tile_grid_shape = tuple(int(width) for width in field_tiles.shape[:3])
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    if tile_grid_shape != mesh_shape:
        raise ValueError(
            "Tiled field communication requires one logical tile per device: "
            f"field tile topology {tile_grid_shape} does not match device mesh {mesh_shape}."
        )


def _validate_vector_tile_topology(field_tiles, mesh):
    if _is_stacked_tiled_vector_field(field_tiles):
        tile_grid_shape = tuple(int(width) for width in field_tiles.shape[1:4])
    else:
        tile_grid_shape = tuple(int(width) for width in field_tiles[0].shape[:3])
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    if tile_grid_shape != mesh_shape:
        raise ValueError(
            "Tiled vector communication requires one logical tile per device: "
            f"field tile topology {tile_grid_shape} does not match device mesh {mesh_shape}."
        )


def _local_refresh_reduced_axis(tile, axis, g, boundary_condition):
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)
    interior = tile[tuple(interior_slice)]

    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)

    tile = tile.at[tuple(lower_slice)].set(jnp.broadcast_to(interior, tile[tuple(lower_slice)].shape))
    tile = tile.at[tuple(upper_slice)].set(jnp.broadcast_to(interior, tile[tuple(upper_slice)].shape))

    if boundary_condition == BC_CONDUCTING:
        tile = tile.at[tuple(lower_slice)].set(0.0)
        tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _local_refresh_scalar_tile(tile, g, boundary_conditions, reduced_axes, send_positive, send_negative):
    bc_x, bc_y, bc_z = boundary_conditions
    reduced_x, reduced_y, reduced_z = reduced_axes

    if reduced_x:
        tile = _local_refresh_reduced_axis(tile, 0, g, bc_x)
    else:
        # Halo refresh: source i sends its +x interior to destination i + 1,
        # so this device receives its lower x ghost from the left neighbor.
        lower_x = jax.lax.ppermute(tile[-2 * g:-g, :, :], "tile_x", send_positive[0])
        # Source i sends its -x interior to destination i - 1, filling the
        # receiver's upper x ghost from the right neighbor.
        upper_x = jax.lax.ppermute(tile[g:2 * g, :, :], "tile_x", send_negative[0])
        tile = tile.at[:g, :, :].set(lower_x)
        tile = tile.at[-g:, :, :].set(upper_x)

    if reduced_y:
        tile = _local_refresh_reduced_axis(tile, 1, g, bc_y)
    else:
        lower_y = jax.lax.ppermute(tile[:, -2 * g:-g, :], "tile_y", send_positive[1])
        upper_y = jax.lax.ppermute(tile[:, g:2 * g, :], "tile_y", send_negative[1])
        tile = tile.at[:, :g, :].set(lower_y)
        tile = tile.at[:, -g:, :].set(upper_y)

    if reduced_z:
        tile = _local_refresh_reduced_axis(tile, 2, g, bc_z)
    else:
        lower_z = jax.lax.ppermute(tile[:, :, -2 * g:-g], "tile_z", send_positive[2])
        upper_z = jax.lax.ppermute(tile[:, :, g:2 * g], "tile_z", send_negative[2])
        tile = tile.at[:, :, :g].set(lower_z)
        tile = tile.at[:, :, -g:].set(upper_z)

    return tile


def _local_fold_reduced_axis(tile, axis, g, boundary_condition):
    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)

    ghost_sum = jnp.sum(tile[tuple(lower_slice)], axis=axis, keepdims=True)
    ghost_sum = ghost_sum + jnp.sum(tile[tuple(upper_slice)], axis=axis, keepdims=True)
    sign = -1.0 if boundary_condition == BC_CONDUCTING else 1.0

    tile = tile.at[tuple(interior_slice)].add(sign * ghost_sum)
    tile = tile.at[tuple(lower_slice)].set(0.0)
    tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _add_exterior_conducting_fold(tile, axis, g, lower_ghost, upper_ghost, axis_name, axis_size):
    lower_index = jax.lax.axis_index(axis_name) == 0
    upper_index = jax.lax.axis_index(axis_name) == axis_size - 1

    lower_target = [slice(None), slice(None), slice(None)]
    lower_target[axis] = slice(g, 2 * g)
    upper_target = [slice(None), slice(None), slice(None)]
    upper_target[axis] = slice(-2 * g, -g)

    tile = jax.lax.cond(
        lower_index,
        lambda local_tile: local_tile.at[tuple(lower_target)].add(-lower_ghost),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        upper_index,
        lambda local_tile: local_tile.at[tuple(upper_target)].add(-upper_ghost),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def _local_fold_scalar_tile(tile, g, boundary_conditions, reduced_axes, mesh_shape, send_positive, send_negative):
    bc_x, bc_y, bc_z = boundary_conditions
    reduced_x, reduced_y, reduced_z = reduced_axes

    if reduced_x:
        tile = _local_fold_reduced_axis(tile, 0, g, bc_x)
    else:
        lower_ghost = tile[:g, :, :]
        upper_ghost = tile[-g:, :, :]
        # Folding reverses halo refresh ownership: a lower ghost belongs to the
        # left neighbor's upper interior, so it is sent in the negative direction.
        from_right = jax.lax.ppermute(lower_ghost, "tile_x", send_negative[0])
        # An upper ghost belongs to the right neighbor's lower interior.
        from_left = jax.lax.ppermute(upper_ghost, "tile_x", send_positive[0])
        tile = tile.at[-2 * g:-g, :, :].add(from_right)
        tile = tile.at[g:2 * g, :, :].add(from_left)
        if bc_x == BC_CONDUCTING:
            tile = _add_exterior_conducting_fold(tile, 0, g, lower_ghost, upper_ghost, "tile_x", mesh_shape[0])
        tile = tile.at[:g, :, :].set(0.0)
        tile = tile.at[-g:, :, :].set(0.0)

    if reduced_y:
        tile = _local_fold_reduced_axis(tile, 1, g, bc_y)
    else:
        lower_ghost = tile[:, :g, :]
        upper_ghost = tile[:, -g:, :]
        from_right = jax.lax.ppermute(lower_ghost, "tile_y", send_negative[1])
        from_left = jax.lax.ppermute(upper_ghost, "tile_y", send_positive[1])
        tile = tile.at[:, -2 * g:-g, :].add(from_right)
        tile = tile.at[:, g:2 * g, :].add(from_left)
        if bc_y == BC_CONDUCTING:
            tile = _add_exterior_conducting_fold(tile, 1, g, lower_ghost, upper_ghost, "tile_y", mesh_shape[1])
        tile = tile.at[:, :g, :].set(0.0)
        tile = tile.at[:, -g:, :].set(0.0)

    if reduced_z:
        tile = _local_fold_reduced_axis(tile, 2, g, bc_z)
    else:
        lower_ghost = tile[:, :, :g]
        upper_ghost = tile[:, :, -g:]
        from_right = jax.lax.ppermute(lower_ghost, "tile_z", send_negative[2])
        from_left = jax.lax.ppermute(upper_ghost, "tile_z", send_positive[2])
        tile = tile.at[:, :, -2 * g:-g].add(from_right)
        tile = tile.at[:, :, g:2 * g].add(from_left)
        if bc_z == BC_CONDUCTING:
            tile = _add_exterior_conducting_fold(tile, 2, g, lower_ghost, upper_ghost, "tile_z", mesh_shape[2])
        tile = tile.at[:, :, :g].set(0.0)
        tile = tile.at[:, :, -g:].set(0.0)

    return tile


def _apply_local_scalar_conducting(tile, g, boundary_conditions, mesh_shape):
    lower = g
    upper = -g - 1
    bc_x, bc_y, bc_z = boundary_conditions

    if bc_x == BC_CONDUCTING:
        ix = jax.lax.axis_index("tile_x")
        tile = jax.lax.cond(ix == 0, lambda f: f.at[lower, :, :].set(0.0), lambda f: f, tile)
        tile = jax.lax.cond(ix == mesh_shape[0] - 1, lambda f: f.at[upper, :, :].set(0.0), lambda f: f, tile)
    if bc_y == BC_CONDUCTING:
        iy = jax.lax.axis_index("tile_y")
        tile = jax.lax.cond(iy == 0, lambda f: f.at[:, lower, :].set(0.0), lambda f: f, tile)
        tile = jax.lax.cond(iy == mesh_shape[1] - 1, lambda f: f.at[:, upper, :].set(0.0), lambda f: f, tile)
    if bc_z == BC_CONDUCTING:
        iz = jax.lax.axis_index("tile_z")
        tile = jax.lax.cond(iz == 0, lambda f: f.at[:, :, lower].set(0.0), lambda f: f, tile)
        tile = jax.lax.cond(iz == mesh_shape[2] - 1, lambda f: f.at[:, :, upper].set(0.0), lambda f: f, tile)

    return tile


def make_distributed_ghost_updater(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_update(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_refresh_scalar_tile(tile, g, boundary_conditions, reduced_axes, send_positive, send_negative)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_update = jax.shard_map(
        local_update,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def update(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_update(field_tiles)

    return update


def make_distributed_vector_ghost_updater(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_update(local_tiles):
        def update_component(local_component):
            tile = local_component[0, 0, 0]
            tile = _local_refresh_scalar_tile(tile, g, boundary_conditions, reduced_axes, send_positive, send_negative)
            return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

        return jax.vmap(update_component, in_axes=0, out_axes=0)(local_tiles)

    mapped_update = jax.shard_map(
        local_update,
        mesh=mesh,
        in_specs=VECTOR_TILE_SPEC,
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )

    def update(field_tiles):
        _validate_vector_tile_topology(field_tiles, mesh)
        stacked_tiles = _stack_tiled_vector_field(field_tiles)
        refreshed = mapped_update(stacked_tiles)
        return _restore_tiled_vector_layout(refreshed, field_tiles)

    return update


def make_distributed_ghost_folder(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_fold(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_fold_scalar_tile(tile, g, boundary_conditions, reduced_axes, mesh_shape, send_positive, send_negative)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_fold = jax.shard_map(
        local_fold,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def fold(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_fold(field_tiles)

    return fold


def make_distributed_vector_ghost_folder(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_fold(local_tiles):
        def fold_component(local_component):
            tile = local_component[0, 0, 0]
            tile = _local_fold_scalar_tile(tile, g, boundary_conditions, reduced_axes, mesh_shape, send_positive, send_negative)
            return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

        return jax.vmap(fold_component, in_axes=0, out_axes=0)(local_tiles)

    mapped_fold = jax.shard_map(
        local_fold,
        mesh=mesh,
        in_specs=VECTOR_TILE_SPEC,
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )

    def fold(field_tiles):
        _validate_vector_tile_topology(field_tiles, mesh)
        stacked_tiles = _stack_tiled_vector_field(field_tiles)
        folded = mapped_fold(stacked_tiles)
        return _restore_tiled_vector_layout(folded, field_tiles)

    return fold


def make_distributed_conducting_bc(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)

    def local_apply(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _apply_local_scalar_conducting(tile, g, boundary_conditions, mesh_shape)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_apply = jax.shard_map(
        local_apply,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def apply(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_apply(field_tiles)

    return apply


def make_distributed_electric_conducting_bc(mesh, tile_shape, boundary_conditions, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)

    def local_apply(local_tiles):
        Ex = local_tiles[0, 0, 0, 0]
        Ey = local_tiles[1, 0, 0, 0]
        Ez = local_tiles[2, 0, 0, 0]
        lower = g
        upper = -g - 1
        bc_x, bc_y, bc_z = boundary_conditions

        if bc_x == BC_CONDUCTING:
            ix = jax.lax.axis_index("tile_x")
            Ey = jax.lax.cond(ix == 0, lambda f: f.at[lower, :, :].set(0.0), lambda f: f, Ey)
            Ez = jax.lax.cond(ix == 0, lambda f: f.at[lower, :, :].set(0.0), lambda f: f, Ez)
            Ey = jax.lax.cond(ix == mesh_shape[0] - 1, lambda f: f.at[upper, :, :].set(0.0), lambda f: f, Ey)
            Ez = jax.lax.cond(ix == mesh_shape[0] - 1, lambda f: f.at[upper, :, :].set(0.0), lambda f: f, Ez)

        if bc_y == BC_CONDUCTING:
            iy = jax.lax.axis_index("tile_y")
            Ex = jax.lax.cond(iy == 0, lambda f: f.at[:, lower, :].set(0.0), lambda f: f, Ex)
            Ez = jax.lax.cond(iy == 0, lambda f: f.at[:, lower, :].set(0.0), lambda f: f, Ez)
            Ex = jax.lax.cond(iy == mesh_shape[1] - 1, lambda f: f.at[:, upper, :].set(0.0), lambda f: f, Ex)
            Ez = jax.lax.cond(iy == mesh_shape[1] - 1, lambda f: f.at[:, upper, :].set(0.0), lambda f: f, Ez)

        if bc_z == BC_CONDUCTING:
            iz = jax.lax.axis_index("tile_z")
            Ex = jax.lax.cond(iz == 0, lambda f: f.at[:, :, lower].set(0.0), lambda f: f, Ex)
            Ey = jax.lax.cond(iz == 0, lambda f: f.at[:, :, lower].set(0.0), lambda f: f, Ey)
            Ex = jax.lax.cond(iz == mesh_shape[2] - 1, lambda f: f.at[:, :, upper].set(0.0), lambda f: f, Ex)
            Ey = jax.lax.cond(iz == mesh_shape[2] - 1, lambda f: f.at[:, :, upper].set(0.0), lambda f: f, Ey)

        stacked = jnp.stack((Ex, Ey, Ez), axis=0)
        return stacked[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_apply = jax.shard_map(
        local_apply,
        mesh=mesh,
        in_specs=VECTOR_TILE_SPEC,
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )

    def apply(field_tiles):
        _validate_vector_tile_topology(field_tiles, mesh)
        stacked_tiles = _stack_tiled_vector_field(field_tiles)
        applied = mapped_apply(stacked_tiles)
        return _restore_tiled_vector_layout(applied, field_tiles)

    return apply


def update_tiled_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh scalar tile halos with one logical tile per JAX device.

    Scalar tiled fields have logical shape
    ``(ntx, nty, ntz, tile_nx + 2*g, tile_ny + 2*g, tile_nz + 2*g)``.
    Cross-tile communication uses ``jax.lax.ppermute`` inside
    ``jax.shard_map`` over the named mesh axes ``tile_x``, ``tile_y``, and
    ``tile_z``.  The leading tile topology must match the device mesh.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, field_tiles)
    updater = make_distributed_ghost_updater(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return updater(field_tiles)


def update_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh tiled vector-field halos, preserving stacked or tuple layout.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, _stack_tiled_vector_field(field_tiles), vector=True)
    updater = make_distributed_vector_ghost_updater(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return updater(field_tiles)


def update_tiled_ghost_cells_for_pml(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh scalar halos with PML-active periodic axes made nonwrapping.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, field_tiles)
    updater = make_distributed_ghost_updater(
        mesh,
        tile_shape,
        _effective_pml_boundary_tuple(world),
        num_guard_cells,
    )
    return updater(field_tiles)


def update_tiled_vector_ghost_cells_for_pml(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Refresh vector halos with PML-active periodic axes made nonwrapping.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, _stack_tiled_vector_field(field_tiles), vector=True)
    updater = make_distributed_vector_ghost_updater(
        mesh,
        tile_shape,
        _effective_pml_boundary_tuple(world),
        num_guard_cells,
    )
    return updater(field_tiles)


def apply_tiled_conducting_bc(E_tiles, world, num_guard_cells=2):
    """
    Zero tangential electric-field components only on global conducting walls.
    """

    tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, _stack_tiled_vector_field(E_tiles), vector=True)
    apply_bc = make_distributed_electric_conducting_bc(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return apply_bc(E_tiles)


def apply_tiled_scalar_conducting_bc(field_tiles, world, num_guard_cells=2):
    """
    Zero scalar field values only on global conducting walls.
    """

    tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, field_tiles)
    apply_bc = make_distributed_conducting_bc(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return apply_bc(field_tiles)


def fold_tiled_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Add tile-ghost deposits into owning interiors, then clear ghost cells.

    Folding uses the same x -> y -> z order as halo refresh.  A ghost deposit
    is sent back to the neighboring interior that owns it; conducting exterior
    deposits reflect only on devices touching the true global walls.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, field_tiles)
    folder = make_distributed_ghost_folder(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return folder(field_tiles)


def fold_tiled_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Fold tile-ghost deposits for a tiled vector field.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    mesh = _mesh_from_world_or_field(world, _stack_tiled_vector_field(field_tiles), vector=True)
    folder = make_distributed_vector_ghost_folder(
        mesh,
        tile_shape,
        _boundary_tuple(world["boundary_conditions"]),
        num_guard_cells,
    )
    return folder(field_tiles)
