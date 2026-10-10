"""Public field and particle boundary adapters over distributed halo primitives."""
import jax
import jax.numpy as jnp

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from .grid_and_stencil import BC_CONDUCTING, BC_CONSTANT
from .halo_exchange import (
    MESH_AXES, SCALAR_TILE_SPEC, VECTOR_TILE_SPEC,
    make_field_mesh,
    make_distributed_ghost_updater, make_distributed_vector_ghost_updater,
    make_distributed_ghost_folder, make_distributed_vector_ghost_folder,
    make_distributed_zero_boundary, make_distributed_constant_boundary,
    _is_stacked_tiled_vector_field,
)
from .sources import particle_vector_reflecting_parity, source_boundaries

BC_TYPE_FIELD = 0
BC_TYPE_PARTICLE = 1


def _reject_field_reflecting_parity(reflecting_parity):
    """Reflection parity describes particle deposits; field walls ignore it."""

    if reflecting_parity is not None:
        raise ValueError("reflecting_parity is only valid for particle boundary conditions.")


def _as_python_int(value):
    return int(jax.device_get(value))


def _boundary_tuple(boundary_conditions):
    if isinstance(boundary_conditions, tuple):
        return tuple(_as_python_int(value) for value in boundary_conditions)
    return (
        _as_python_int(boundary_conditions["x"]),
        _as_python_int(boundary_conditions["y"]),
        _as_python_int(boundary_conditions["z"]),
    )


def _boundary_conditions_for_type(static_parameters, bc_type):
    bc_type = int(bc_type)
    if bc_type == BC_TYPE_FIELD:
        return _boundary_tuple(static_parameters.boundary_conditions)
    raise ValueError("bc_type must be 0 for field boundaries or 1 for particle boundaries.")


def update_tiled_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    """
    Refresh scalar tile halos with one logical tile per JAX device.

    Scalar tiled fields have logical shape
    ``(ntx, nty, ntz, tile_nx + 2*g, tile_ny + 2*g, tile_nz + 2*g)``.
    Cross-tile communication uses ``jax.lax.ppermute`` inside
    ``jax.shard_map`` over the named mesh axes ``tile_x``, ``tile_y``, and
    ``tile_z``. The leading tile topology must match the device mesh. Particle
    boundaries mirror exterior halos using ``reflecting_parity``; scalar
    particle deposits are even by default. ``location`` preserves owned upper
    C endpoints at conducting walls. ``preserve_exterior`` keeps computed
    nonperiodic exterior slabs (a bool, or an x/y/z tuple of bools).
    """

    if bc_type == BC_TYPE_PARTICLE:
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=False, vector=False, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    _reject_field_reflecting_parity(reflecting_parity)
    updater = make_distributed_ghost_updater(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
        location=location,
        preserve_exterior=preserve_exterior,
    )
    result = updater(field_tiles)
    return result


def update_tiled_vector_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
    locations=None,
    preserve_exterior=False,
):
    """
    Refresh tiled multi-component field halos, preserving stacked or tuple layout.

    Production vector fields use three components. The component axis remains
    general so related spatial operators can batch several derivative channels
    through the same distributed halo exchange. Particle vectors default to
    odd normal and even tangential reflection parity.
    """

    if bc_type == BC_TYPE_PARTICLE:
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=False, vector=True, reflecting_parity=reflecting_parity)

    _reject_field_reflecting_parity(reflecting_parity)
    # Field boundaries only: conducting walls with staggered locations, or
    # preserved exterior slabs, refresh each component at its own location.
    preserve_any = any(preserve_exterior) if isinstance(preserve_exterior, tuple) else preserve_exterior
    if (preserve_any or locations is not None and
            BC_CONDUCTING in _boundary_conditions_for_type(static_parameters, bc_type)):
        locations = (None,) * len(field_tiles) if locations is None else locations
        result = tuple(update_tiled_ghost_cells(
            value, static_parameters, num_guard_cells, bc_type,
            location=location, preserve_exterior=preserve_exterior)
            for value, location in zip(field_tiles, locations))
        return jnp.stack(result) if _is_stacked_tiled_vector_field(field_tiles) else result

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    updater = make_distributed_vector_ghost_updater(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
    )
    result = updater(field_tiles)
    return result


def apply_tiled_zero_boundary(field_tiles, static_parameters, axis, num_guard_cells=2, *, location=None):
    """
    Zero scalar values on the global conducting wall for one spatial axis.
    """

    axis = int(axis)
    boundary_conditions = _boundary_tuple(static_parameters.boundary_conditions)
    if boundary_conditions[axis] != BC_CONDUCTING:
        return update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells, location=location)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    apply_bc = make_distributed_zero_boundary(
        mesh,
        tile_shape,
        axis,
        num_guard_cells,
    )
    field_tiles = apply_bc(field_tiles)
    return update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells, location=location)


def apply_tiled_pec_boundary(fields, static_parameters):
    """Yee tangential electric clamp at the physical endpoints, g and g+n."""
    result = list(fields)
    for axis, bc in enumerate(static_parameters.boundary_conditions):
        if bc == BC_CONDUCTING:
            for i in range(3):
                if i != axis:
                    result[i] = apply_tiled_zero_boundary(
                        result[i], static_parameters, axis, static_parameters.guard_cells,
                        location=D_FIELD_LOCATIONS[i])
    return tuple(result)


def apply_tiled_constant_boundary(field_tiles, static_parameters, axis, num_guard_cells=2):
    """
    Fill exterior ghost cells from the adjacent interior plane.

    This is used by the electrostatic scalar potential on conducting walls and
    by explicit constant field boundaries.  Internal tile halos are still
    refreshed through the distributed ppermute path before the exterior ghosts
    are overwritten.
    """

    axis = int(axis)
    field_tiles = update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells)

    boundary_conditions = _boundary_tuple(static_parameters.boundary_conditions)
    if boundary_conditions[axis] not in (BC_CONDUCTING, BC_CONSTANT):
        return field_tiles

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    apply_bc = make_distributed_constant_boundary(
        mesh,
        tile_shape,
        axis,
        num_guard_cells,
    )
    return apply_bc(field_tiles)


def fold_tiled_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
):
    """
    Add tile-ghost deposits into owning interiors, then clear ghost cells.

    Folding uses the same x -> y -> z order as halo refresh. A ghost deposit
    is sent back to the neighboring interior that owns it; conducting exterior
    deposits reflect only on devices touching the true global walls. Particle
    deposits use the nodal method of images about the wall nodes, the same
    for every solver: charge sits on collocated (C) nodes and currents on the
    Yee E-component locations.
    """

    if bc_type == BC_TYPE_PARTICLE:
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=True, vector=False, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    _reject_field_reflecting_parity(reflecting_parity)
    folder = make_distributed_ghost_folder(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
    )
    result = folder(field_tiles)
    return result

def fold_tiled_vector_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
):
    """
    Fold tile-ghost deposits for a tiled vector field.

    Particle currents fold at the Yee E-component locations with odd normal
    and even tangential parity (see ``fold_tiled_ghost_cells``).
    """

    if bc_type == BC_TYPE_PARTICLE:
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=True, vector=True, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    _reject_field_reflecting_parity(reflecting_parity)
    folder = make_distributed_vector_ghost_folder(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
    )
    result = folder(field_tiles)
    return result
