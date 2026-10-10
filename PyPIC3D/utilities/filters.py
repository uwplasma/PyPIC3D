from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.sharding import PartitionSpec as P

from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_FIELD,
    SCALAR_TILE_SPEC,
    VECTOR_TILE_SPEC,
    update_tiled_ghost_cells,
    update_tiled_vector_ghost_cells,
)


def _active_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g, -g)


def _backward_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g - 1, -g - 1)


def _forward_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g + 1, None if g == 1 else -g + 1)


def _stencil_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g - 1, None if g == 1 else -g + 1)


def _is_stacked_vector_field(field):
    return hasattr(field, "ndim") and field.ndim >= 4 and int(field.shape[0]) == 3


def _stack_vector_field(field):
    if _is_stacked_vector_field(field):
        return field
    return jnp.stack(field, axis=0)


def _restore_vector_field(stacked_field, original_field):
    if _is_stacked_vector_field(original_field):
        return stacked_field
    return stacked_field[0], stacked_field[1], stacked_field[2]


@partial(jit, static_argnames=("num_guard_cells",))
def bilinear_filter(phi, num_guard_cells=1):
    """
    Apply a 3D tri-linear smoothing filter to a ghost-celled field.

    The last three axes are spatial. ``num_guard_cells`` selects the physical
    interior ``g:-g``. The one-cell stencil is read directly from the guard
    cells, and all guard cells are left unchanged.
    """

    stencil = _stencil_slice(num_guard_cells)
    stencil_values = phi[..., stencil, stencil, stencil]

    # Apply the separable [1, 2, 1] / 4 stencil along each spatial axis.
    filtered_x = (
        stencil_values[..., :-2, :, :]
        + 2.0 * stencil_values[..., 1:-1, :, :]
        + stencil_values[..., 2:, :, :]
    ) / 4.0
    filtered_xy = (
        filtered_x[..., :, :-2, :]
        + 2.0 * filtered_x[..., :, 1:-1, :]
        + filtered_x[..., :, 2:, :]
    ) / 4.0
    filtered = (
        filtered_xy[..., :, :, :-2]
        + 2.0 * filtered_xy[..., :, :, 1:-1]
        + filtered_xy[..., :, :, 2:]
    ) / 4.0

    active = _active_slice(num_guard_cells)
    return phi.at[..., active, active, active].set(filtered)


@partial(jit, static_argnames=("num_guard_cells",))
def digital_filter(phi, alpha, num_guard_cells=1):
    """
    Apply a 3D nearest-neighbor digital filter to a ghost-celled field.

    The last three axes are spatial. ``num_guard_cells`` selects the physical
    interior ``g:-g``. The six face neighbors are read directly from the guard
    cells, and all guard cells are left unchanged.
    """

    active = _active_slice(num_guard_cells)
    backward = _backward_slice(num_guard_cells)
    forward = _forward_slice(num_guard_cells)

    center = phi[..., active, active, active]
    neighbors = (
        phi[..., backward, active, active]
        + phi[..., forward, active, active]
        + phi[..., active, backward, active]
        + phi[..., active, forward, active]
        + phi[..., active, active, backward]
        + phi[..., active, active, forward]
    )

    neighbor_weight = (1.0 - alpha) / 6.0
    filtered = alpha * center + neighbor_weight * neighbors

    return phi.at[..., active, active, active].set(filtered)


def digital_filter_vector(field, alpha, num_guard_cells=1):
    """Apply the six-neighbor digital filter component-wise to a vector field."""

    stacked = _stack_vector_field(field)
    filtered = vmap(
        lambda component: digital_filter(component, alpha, num_guard_cells=num_guard_cells),
        in_axes=0,
        out_axes=0,
    )(stacked)
    return _restore_vector_field(filtered, field)


def _tiled_scalar_filter(field_tiles, static_parameters, filter_function, filter_args, bc_type):
    g = int(static_parameters.guard_cells)
    mesh = static_parameters.field_mesh

    field_tiles = update_tiled_ghost_cells(
        field_tiles,
        static_parameters,
        num_guard_cells=g,
        bc_type=bc_type,
    )

    def filter_local_tile(local_tiles, *local_filter_args):
        tile = local_tiles[0, 0, 0]
        tile = filter_function(tile, *local_filter_args, num_guard_cells=g)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_filter = jax.shard_map(
        filter_local_tile,
        mesh=mesh,
        in_specs=(SCALAR_TILE_SPEC,) + tuple(P() for _ in filter_args),
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )
    field_tiles = mapped_filter(field_tiles, *filter_args)

    return update_tiled_ghost_cells(
        field_tiles,
        static_parameters,
        num_guard_cells=g,
        bc_type=bc_type,
    )


def _tiled_vector_filter(field_tiles, static_parameters, filter_function, filter_args, bc_type):
    g = int(static_parameters.guard_cells)
    mesh = static_parameters.field_mesh

    field_tiles = update_tiled_vector_ghost_cells(
        field_tiles,
        static_parameters,
        num_guard_cells=g,
        bc_type=bc_type,
    )
    stacked_tiles = _stack_vector_field(field_tiles)

    def filter_local_tile(local_tiles, *local_filter_args):
        local_components = local_tiles[:, 0, 0, 0]
        filtered_components = vmap(
            lambda component: filter_function(
                component,
                *local_filter_args,
                num_guard_cells=g,
            ),
            in_axes=0,
            out_axes=0,
        )(local_components)
        return filtered_components[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_filter = jax.shard_map(
        filter_local_tile,
        mesh=mesh,
        in_specs=(VECTOR_TILE_SPEC,) + tuple(P() for _ in filter_args),
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )
    filtered_tiles = mapped_filter(stacked_tiles, *filter_args)
    field_tiles = _restore_vector_field(filtered_tiles, field_tiles)

    return update_tiled_vector_ghost_cells(
        field_tiles,
        static_parameters,
        num_guard_cells=g,
        bc_type=bc_type,
    )


def tiled_bilinear_filter(field_tiles, static_parameters, bc_type=BC_TYPE_FIELD):
    """Refresh, filter, and refresh a distributed scalar tiled field."""

    return _tiled_scalar_filter(
        field_tiles,
        static_parameters,
        bilinear_filter,
        (),
        bc_type,
    )


def tiled_digital_filter(field_tiles, alpha, static_parameters, bc_type=BC_TYPE_FIELD):
    """Refresh, digitally filter, and refresh a distributed scalar tiled field."""

    return _tiled_scalar_filter(
        field_tiles,
        static_parameters,
        digital_filter,
        (alpha,),
        bc_type,
    )


def tiled_bilinear_filter_vector(field_tiles, static_parameters, bc_type=BC_TYPE_FIELD):
    """Refresh, filter, and refresh a distributed tiled vector field."""

    return _tiled_vector_filter(
        field_tiles,
        static_parameters,
        bilinear_filter,
        (),
        bc_type,
    )


def tiled_digital_filter_vector(field_tiles, alpha, static_parameters, bc_type=BC_TYPE_FIELD):
    """Refresh, digitally filter, and refresh a distributed tiled vector field."""

    return _tiled_vector_filter(
        field_tiles,
        static_parameters,
        digital_filter,
        (alpha,),
        bc_type,
    )
