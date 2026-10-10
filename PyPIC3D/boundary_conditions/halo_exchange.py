"""Distributed halo exchange and additive folding with explicit C/V ownership.

These primitives consume mesh, staggering, and parity directly. Solver and
source policies belong to the public adapters in ghost_cells and sources.
"""
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .grid_and_stencil import BC_ABSORBING, BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from .ownership import staggered_mirror

MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None, None)
VECTOR_TILE_SPEC = P(None, "tile_x", "tile_y", "tile_z", None, None, None)


def _reduced_axes_from_tile_shape(tile_shape, mesh_shape):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = [int(width) for width in mesh_shape]
    return (
        tile_nx == 1 and ntx == 1,
        tile_ny == 1 and nty == 1,
        tile_nz == 1 and ntz == 1,
    )


def _is_stacked_tiled_vector_field(field_tiles):
    return hasattr(field_tiles, "ndim") and field_tiles.ndim == 7


def _stack_tiled_vector_field(field_tiles):
    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles
    return jnp.stack(field_tiles, axis=0)


def _unstack_tiled_vector_field(field_tiles):
    return tuple(field_tiles[i] for i in range(int(field_tiles.shape[0])))


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


def make_field_mesh(tile_grid_shape):
    """
    Build the JAX device mesh for one logical field tile per device.
    """

    tile_grid_shape = tuple(int(width) for width in tile_grid_shape)
    n_devices = math.prod(tile_grid_shape)
    devices = jax.devices()
    if len(devices) < n_devices:
        raise ValueError(
            "Tiled field communication requires one logical tile per device: "
            f"tile topology {tile_grid_shape} needs {n_devices} devices, "
            f"but JAX exposes {len(devices)}."
        )
    return jax.make_mesh(
        tile_grid_shape,
        MESH_AXES,
        devices=devices[:n_devices],
        axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_AXES),
    )


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


def _local_refresh_reduced_axis(
    tile,
    axis,
    g,
    boundary_condition,
):
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)
    interior = tile[tuple(interior_slice)]

    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)

    if boundary_condition == BC_PERIODIC:
        tile = tile.at[tuple(lower_slice)].set(jnp.broadcast_to(interior, tile[tuple(lower_slice)].shape))
        tile = tile.at[tuple(upper_slice)].set(jnp.broadcast_to(interior, tile[tuple(upper_slice)].shape))
    elif boundary_condition == BC_CONSTANT:
        tile = tile.at[tuple(lower_slice)].set(jnp.broadcast_to(interior, tile[tuple(lower_slice)].shape))
        tile = tile.at[tuple(upper_slice)].set(jnp.broadcast_to(interior, tile[tuple(upper_slice)].shape))
    else:
        tile = tile.at[tuple(lower_slice)].set(0.0)
        tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _axis_slices(axis, g):
    lower_ghost = [slice(None), slice(None), slice(None)]
    upper_ghost = [slice(None), slice(None), slice(None)]
    lower_interior = [slice(None), slice(None), slice(None)]
    upper_interior = [slice(None), slice(None), slice(None)]

    lower_ghost[axis] = slice(0, g)
    upper_ghost[axis] = slice(-g, None)
    lower_interior[axis] = slice(g, 2 * g)
    upper_interior[axis] = slice(-2 * g, -g)

    return (
        tuple(lower_ghost),
        tuple(upper_ghost),
        tuple(lower_interior),
        tuple(upper_interior),
    )


def _refresh_axis(tile, axis, g, axis_name, send_positive, send_negative):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_slices(axis, g)

    # Source i sends its upper interior to destination i + 1, filling the
    # receiver's lower ghost.  With a one-device periodic axis the permutation
    # is ((0, 0),), so this is the same self-exchange used on larger meshes.
    lower_values = jax.lax.ppermute(tile[upper_interior], axis_name, send_positive)
    # Source i sends its lower interior to destination i - 1, filling the
    # receiver's upper ghost.  Empty nonperiodic permutations naturally produce
    # zeros where no neighbor exists.
    upper_values = jax.lax.ppermute(tile[lower_interior], axis_name, send_negative)

    tile = tile.at[lower_ghost].set(lower_values)
    tile = tile.at[upper_ghost].set(upper_values)

    return tile


def _staggered_conducting_axis(tile, axis, g, axis_name, axis_size,
                                positive, negative, location, parity, *, fold):
    """Reflect about actual endpoint nodes, preserving the owned upper C plane.

    Folding creates the even/odd image source. An even source on a wall node
    receives its coincident image, so its point density is doubled; integrating
    nodal densities uses endpoint trapezoid weights. No metric volumes enter.
    """
    n = tile.shape[axis] - 2*g
    staggered = location == 'V'
    upper = g+n-1 if staggered else g+n
    index = jax.lax.axis_index(axis_name)
    parity = 1 if parity is None else parity
    def plane(i):
        result = [slice(None)] * 3
        result[axis] = i
        return tuple(result)
    def wall(a, low):
        outside = range(g) if low else range(upper+1, a.shape[axis])
        original = a
        for i in outside:
            j = i-g
            owner = (-j-1 if staggered else -j) if low else (2*n-1-j if staggered else 2*n-j)
            sign = parity
            # A narrow single-tile direction can fit fewer cells than halos.
            # Continue reflecting until the owner is inside the domain.
            if axis_size == 1:
                owner, walls = staggered_mirror(owner, n, location)
                sign *= parity**len(walls)
            if fold:
                a = a.at[plane(g+owner)].add(sign*original[plane(i)])
                a = a.at[plane(i)].set(0.)
            else:
                a = a.at[plane(i)].set(sign*original[plane(g+owner)])
        return a
    if fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
        if not staggered:
            tile = jax.lax.cond(index == 0,
                lambda a: a.at[plane(g)].multiply(1+parity), lambda a: a, tile)
            tile = jax.lax.cond(index == axis_size-1,
                lambda a: a.at[plane(g+n)].multiply(1+parity), lambda a: a, tile)
        endpoint = tile[plane(g+n)]
        # Reflection has already folded the exterior sources. Exchange only
        # inter-tile deposits, using the ordinary nonperiodic communication.
        tile = _fold_axis(tile, axis, g, axis_name, axis_size, BC_ABSORBING, positive, negative)
    else:
        endpoint = tile[plane(g+n)]
        tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
    if not staggered:
        tile = jax.lax.cond(index == axis_size-1,
                            lambda a: a.at[plane(g+n)].set(endpoint), lambda a: a, tile)
    if not fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
    return tile


def _local_refresh_scalar_tile(
    tile,
    g,
    boundary_conditions,
    reduced_axes,
    mesh_shape,
    send_positive,
    send_negative,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    axis_parities = (None, None, None) if reflecting_parity is None else reflecting_parity
    for axis, axis_name, boundary_condition, reduced_axis, positive, negative, parity in zip(
        range(3),
        MESH_AXES,
        boundary_conditions,
        reduced_axes,
        send_positive,
        send_negative,
        axis_parities,
    ):
        if boundary_condition == BC_CONDUCTING and location is not None and parity is not None:
            tile = _staggered_conducting_axis(
                tile, axis, g, axis_name, mesh_shape[axis], positive, negative,
                location[axis], parity, fold=False)
            continue
        keep_exterior = (preserve_exterior[axis] if isinstance(preserve_exterior, tuple)
                         else preserve_exterior)
        if (keep_exterior and boundary_condition != BC_PERIODIC or
                boundary_condition == BC_CONDUCTING and location is not None):
            # C endpoints are owned by the final tile. Restore them before
            # transverse exchange, along with computed exterior values when
            # requested by the metric constitutive/projector operators.
            lower, upper, _, _ = _axis_slices(axis, g)
            before = tile
            tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
            index = jax.lax.axis_index(axis_name)
            if keep_exterior:
                tile = jax.lax.cond(index == 0,
                    lambda a: a.at[lower].set(before[lower]), lambda a: a, tile)
                tile = jax.lax.cond(index == mesh_shape[axis]-1,
                    lambda a: a.at[upper].set(before[upper]), lambda a: a, tile)
            elif location[axis] == 'C':
                face = _axis_boundary_plane(axis, -g)
                tile = jax.lax.cond(index == mesh_shape[axis]-1,
                    lambda a: a.at[face].set(before[face]), lambda a: a, tile)
            continue
        # A normal staggered source owns the lower absorbing face even though
        # it is stored in a halo. Preserve it before the transverse directions
        # are processed, so corner sources receive their transverse folding.
        keep_face = (boundary_condition == BC_ABSORBING and location is not None
                     and parity is not None and location[axis] == 'V')
        if keep_face:
            face = _axis_boundary_plane(axis, g-1)
            lower_flux = tile[face]
        if reduced_axis:
            tile = _local_refresh_reduced_axis(
                tile,
                axis,
                g,
                boundary_condition,
            )
        else:
            tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
            if boundary_condition == BC_CONSTANT:
                tile = _apply_local_constant_boundary_axis(tile, axis, g, axis_name, mesh_shape[axis])
        if keep_face:
            tile = jax.lax.cond(jax.lax.axis_index(axis_name) == 0,
                lambda a: a.at[face].set(lower_flux), lambda a: a, tile)

    return tile


def _local_fold_reduced_axis(
    tile,
    axis,
    g,
    boundary_condition,
):
    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)

    ghost_sum = jnp.sum(tile[tuple(lower_slice)], axis=axis, keepdims=True)
    ghost_sum = ghost_sum + jnp.sum(tile[tuple(upper_slice)], axis=axis, keepdims=True)
    if boundary_condition == BC_PERIODIC:
        tile = tile.at[tuple(interior_slice)].add(ghost_sum)
    elif boundary_condition == BC_CONDUCTING:
        tile = tile.at[tuple(interior_slice)].add(-ghost_sum)
    tile = tile.at[tuple(lower_slice)].set(0.0)
    tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _add_exterior_boundary_fold(
    tile,
    axis,
    g,
    lower_ghost,
    upper_ghost,
    axis_name,
    axis_size,
):
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


def _fold_axis(
    tile,
    axis,
    g,
    axis_name,
    axis_size,
    boundary_condition,
    send_positive,
    send_negative,
):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_slices(axis, g)

    lower_values = tile[lower_ghost]
    upper_values = tile[upper_ghost]
    # Folding reverses halo refresh ownership: a lower ghost belongs to the
    # negative neighbor's upper interior, and an upper ghost belongs to the
    # positive neighbor's lower interior.
    from_positive_neighbor = jax.lax.ppermute(lower_values, axis_name, send_negative)
    from_negative_neighbor = jax.lax.ppermute(upper_values, axis_name, send_positive)

    tile = tile.at[upper_interior].add(from_positive_neighbor)
    tile = tile.at[lower_interior].add(from_negative_neighbor)
    if boundary_condition == BC_CONDUCTING:
        tile = _add_exterior_boundary_fold(
            tile,
            axis,
            g,
            lower_values,
            upper_values,
            axis_name,
            axis_size,
        )
    tile = tile.at[lower_ghost].set(0.0)
    tile = tile.at[upper_ghost].set(0.0)

    return tile


def _local_fold_scalar_tile(
    tile,
    g,
    boundary_conditions,
    reduced_axes,
    mesh_shape,
    send_positive,
    send_negative,
    reflecting_parity=None,
    location=None,
):
    axis_parities = (None, None, None) if reflecting_parity is None else reflecting_parity
    for axis, axis_name, axis_size, boundary_condition, reduced_axis, positive, negative, parity in zip(
        range(3),
        MESH_AXES,
        mesh_shape,
        boundary_conditions,
        reduced_axes,
        send_positive,
        send_negative,
        axis_parities,
    ):
        if boundary_condition == BC_CONDUCTING and location is not None:
            tile = _staggered_conducting_axis(
                tile, axis, g, axis_name, mesh_shape[axis], positive, negative,
                location[axis], parity, fold=True)
            continue
        keep_face = (boundary_condition == BC_ABSORBING and location is not None
                     and parity is not None and location[axis] == 'V')
        if keep_face:
            face = _axis_boundary_plane(axis, g-1)
            lower_flux = tile[face]
        if reduced_axis:
            tile = _local_fold_reduced_axis(
                tile,
                axis,
                g,
                boundary_condition,
            )
        else:
            tile = _fold_axis(
                tile,
                axis,
                g,
                axis_name,
                axis_size,
                boundary_condition,
                positive,
                negative,
            )
        if keep_face:
            tile = jax.lax.cond(jax.lax.axis_index(axis_name) == 0,
                lambda a: a.at[face].set(lower_flux), lambda a: a, tile)

    return tile


def _axis_boundary_plane(axis, index):
    plane = [slice(None), slice(None), slice(None)]
    plane[axis] = index
    return tuple(plane)


def _axis_constant_boundary_slices(axis, g):
    lower_ghost = [slice(None), slice(None), slice(None)]
    upper_ghost = [slice(None), slice(None), slice(None)]
    lower_interior = [slice(None), slice(None), slice(None)]
    upper_interior = [slice(None), slice(None), slice(None)]

    lower_ghost[axis] = slice(0, g)
    upper_ghost[axis] = slice(-g, None)
    lower_interior[axis] = slice(g, g + 1)
    upper_interior[axis] = slice(-g - 1, -g)

    return (
        tuple(lower_ghost),
        tuple(upper_ghost),
        tuple(lower_interior),
        tuple(upper_interior),
    )


def _apply_local_zero_boundary_axis(tile, axis, g, axis_name, axis_size):
    lower_plane = _axis_boundary_plane(axis, g)
    upper_plane = _axis_boundary_plane(axis, -g)
    tile_index = jax.lax.axis_index(axis_name)

    tile = jax.lax.cond(
        tile_index == 0,
        lambda local_tile: local_tile.at[lower_plane].set(0.0),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        tile_index == axis_size - 1,
        lambda local_tile: local_tile.at[upper_plane].set(0.0),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def _apply_local_constant_boundary_axis(tile, axis, g, axis_name, axis_size):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_constant_boundary_slices(axis, g)
    tile_index = jax.lax.axis_index(axis_name)

    tile = jax.lax.cond(
        tile_index == 0,
        lambda local_tile: local_tile.at[lower_ghost].set(
            jnp.broadcast_to(local_tile[lower_interior], local_tile[lower_ghost].shape)
        ),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        tile_index == axis_size - 1,
        lambda local_tile: local_tile.at[upper_ghost].set(
            jnp.broadcast_to(local_tile[upper_interior], local_tile[upper_ghost].shape)
        ),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def make_distributed_ghost_updater(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    """
    Build a shard-mapped scalar halo refresher.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned updater instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(int(value) for value in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_update(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_refresh_scalar_tile(
            tile,
            g,
            boundary_conditions,
            reduced_axes,
            mesh_shape,
            send_positive,
            send_negative,
            reflecting_parity=reflecting_parity,
            location=location,
            preserve_exterior=preserve_exterior,
        )
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


def make_distributed_vector_ghost_updater(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
):
    """
    Build a shard-mapped vector halo refresher.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned updater instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_update(local_tiles):
        def update_component(local_component):
            tile = local_component[0, 0, 0]
            tile = _local_refresh_scalar_tile(
                tile,
                g,
                boundary_conditions,
                reduced_axes,
                mesh_shape,
                send_positive,
                send_negative,
            )
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


def make_distributed_ghost_folder(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
    location=None,
):
    """
    Build a shard-mapped scalar ghost-deposit folder.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned folder instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(int(value) for value in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_fold(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_fold_scalar_tile(
            tile,
            g,
            boundary_conditions,
            reduced_axes,
            mesh_shape,
            send_positive,
            send_negative,
            reflecting_parity=reflecting_parity,
            location=location,
        )
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


def make_distributed_vector_ghost_folder(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
):
    """
    Build a shard-mapped vector ghost-deposit folder.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned folder instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_fold(local_tiles):
        def fold_component(local_component):
            tile = local_component[0, 0, 0]
            tile = _local_fold_scalar_tile(
                tile,
                g,
                boundary_conditions,
                reduced_axes,
                mesh_shape,
                send_positive,
                send_negative,
            )
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


def make_distributed_zero_boundary(mesh, tile_shape, axis, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    axis = int(axis)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    axis_name = MESH_AXES[axis]
    axis_size = mesh_shape[axis]

    def local_apply(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _apply_local_zero_boundary_axis(tile, axis, g, axis_name, axis_size)
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


def make_distributed_constant_boundary(mesh, tile_shape, axis, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    axis = int(axis)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    axis_name = MESH_AXES[axis]
    axis_size = mesh_shape[axis]

    def local_apply(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _apply_local_constant_boundary_axis(tile, axis, g, axis_name, axis_size)
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


