"""Tiled ghost cells: index geometry, node ownership, transport, and policy.

One logical tile lives on each JAX device. A tiled scalar field has shape
``(ntx, nty, ntz, tile_nx + 2g, tile_ny + 2g, tile_nz + 2g)``; a tiled vector
field is either a stacked seven-dimensional array or a sequence of such
scalars. Cross-tile communication uses ``jax.lax.ppermute`` inside
``jax.shard_map`` over the named mesh axes ``tile_x``, ``tile_y``, ``tile_z``,
and the leading tile topology must match the device mesh.

Refresh copies owner-interior values into neighboring ghost cells; folding
adds ghost deposits back into the interiors that own them and then clears the
ghosts. Both sweep axes in x -> y -> z order. Field walls are pure transport;
particle walls add the nodal method of images about the wall nodes.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from .grid_and_stencil import BC_ABSORBING, BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC

MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None, None)
VECTOR_TILE_SPEC = P(None, "tile_x", "tile_y", "tile_z", None, None, None)

BC_TYPE_FIELD = 0
BC_TYPE_PARTICLE = 1


# ---------------------------------------------------------------------------
# Index geometry and node ownership
# ---------------------------------------------------------------------------

def _plane(axis, index):
    """Index one physical-axis plane or slab of a tile-local scalar."""
    result = [slice(None)] * 3
    result[axis] = index
    return tuple(result)


def _halo_slices(axis, g):
    """Ghost and owner-interior slabs of width ``g`` at both ends of an axis."""
    return (
        _plane(axis, slice(0, g)),
        _plane(axis, slice(-g, None)),
        _plane(axis, slice(g, 2 * g)),
        _plane(axis, slice(-2 * g, -g)),
    )


def boundary_plane(axis, tile, node):
    """Index one physical-axis plane in a six-dimensional tiled scalar."""
    index = [slice(None)] * 6
    index[axis], index[axis + 3] = tile, node
    return tuple(index)


def face_mask(shape, axis, width, guard):
    """Both physical endpoint planes, broadcast over the tiled scalar shape."""
    tile = jnp.arange(shape[axis])
    node = jnp.arange(shape[axis + 3])
    lower = (tile[:, None] == 0) & (node[None, :] == guard)
    upper = (tile[:, None] == shape[axis] - 1) & (node[None, :] == guard + width)
    dimensions = [1] * 6
    dimensions[axis], dimensions[axis + 3] = shape[axis], shape[axis + 3]
    return (lower | upper).reshape(dimensions)


def owned_nodes(shape, location, static):
    """Active nodes, including conducting upper C endpoints owned by a tile."""
    mask = True
    guard = static.guard_cells
    for axis, width in enumerate(static.tile_shape):
        nodes = jnp.arange(shape[axis + 3])
        dimensions = [1] * 6
        dimensions[axis + 3] = len(nodes)
        active = ((nodes >= guard) & (nodes < guard + width)).reshape(dimensions)
        if static.boundary_conditions[axis] == BC_CONDUCTING and location[axis] == 'C':
            active = active | face_mask(shape, axis, width, guard)
        mask = mask & active
    return mask


def staggered_mirror(node, width, location):
    """Return the owner and ordered wall encounters for a C/V image node."""
    vertex = location == 'V'
    last = width - 1 if vertex else width
    walls = []
    while node < 0 or node > last:
        wall = 0 if node < 0 else width
        walls.append(wall)
        node = 2 * wall - node - int(vertex)
    return node, tuple(walls)


# ---------------------------------------------------------------------------
# Device mesh and neighbor permutations
# ---------------------------------------------------------------------------

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


def axis_permutations(axis_size, boundary_condition):
    """
    Return the ``(positive, negative)`` ppermute pairs for one mesh axis.

    The positive permutation sends each source to its ``+1`` neighbor and the
    negative permutation to its ``-1`` neighbor. A periodic axis wraps, so a
    one-device axis yields the self-exchange ``((0, 0),)``. A nonperiodic axis
    omits the pair that would leave the mesh, and ``ppermute`` then delivers
    zeros where no neighbor exists.
    """

    axis_size = int(axis_size)
    if boundary_condition == BC_PERIODIC:
        return (
            tuple((i, (i + 1) % axis_size) for i in range(axis_size)),
            tuple((i, (i - 1) % axis_size) for i in range(axis_size)),
        )
    return (
        tuple((i, i + 1) for i in range(axis_size - 1)),
        tuple((i, i - 1) for i in range(1, axis_size)),
    )


# ---------------------------------------------------------------------------
# Tile-local axis kernels
# ---------------------------------------------------------------------------

def _exchange_axis(tile, axis, g, axis_name, axis_size, boundary_condition,
                   positive, negative, fold):
    """Refresh or fold one axis against its mesh neighbors.

    Refresh sends source ``i``'s upper interior to ``i + 1``, filling the
    receiver's lower ghost, and its lower interior to ``i - 1``. Folding
    reverses that ownership: a lower ghost belongs to the negative neighbor's
    upper interior. Conducting global walls reflect their own exterior
    deposits with a negative sign before the ghosts are cleared.

    Both branches read every source slab from the unmodified tile before any
    write. The slabs overlap when ``2*g`` exceeds the tile width, so writing
    between reads would change the result.
    """

    lower_ghost, upper_ghost, lower_interior, upper_interior = _halo_slices(axis, g)

    if not fold:
        lower_values = jax.lax.ppermute(tile[upper_interior], axis_name, positive)
        upper_values = jax.lax.ppermute(tile[lower_interior], axis_name, negative)
        tile = tile.at[lower_ghost].set(lower_values)
        return tile.at[upper_ghost].set(upper_values)

    lower_values = tile[lower_ghost]
    upper_values = tile[upper_ghost]
    from_positive_neighbor = jax.lax.ppermute(lower_values, axis_name, negative)
    from_negative_neighbor = jax.lax.ppermute(upper_values, axis_name, positive)

    tile = tile.at[upper_interior].add(from_positive_neighbor)
    tile = tile.at[lower_interior].add(from_negative_neighbor)
    if boundary_condition == BC_CONDUCTING:
        index = jax.lax.axis_index(axis_name)
        tile = jax.lax.cond(index == 0,
            lambda a: a.at[lower_interior].add(-lower_values), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size - 1,
            lambda a: a.at[upper_interior].add(-upper_values), lambda a: a, tile)
    tile = tile.at[lower_ghost].set(0.0)
    return tile.at[upper_ghost].set(0.0)


def _reduced_axis(tile, axis, g, boundary_condition, fold):
    """Handle an axis that is one physical cell wide on a one-tile mesh."""

    lower_ghost, upper_ghost, _, _ = _halo_slices(axis, g)
    interior = _plane(axis, slice(g, g + 1))

    if fold:
        ghost_sum = jnp.sum(tile[lower_ghost], axis=axis, keepdims=True)
        ghost_sum = ghost_sum + jnp.sum(tile[upper_ghost], axis=axis, keepdims=True)
        if boundary_condition == BC_PERIODIC:
            tile = tile.at[interior].add(ghost_sum)
        elif boundary_condition == BC_CONDUCTING:
            tile = tile.at[interior].add(-ghost_sum)
        tile = tile.at[lower_ghost].set(0.0)
        return tile.at[upper_ghost].set(0.0)

    if boundary_condition in (BC_PERIODIC, BC_CONSTANT):
        values = tile[interior]
        tile = tile.at[lower_ghost].set(jnp.broadcast_to(values, tile[lower_ghost].shape))
        return tile.at[upper_ghost].set(jnp.broadcast_to(values, tile[upper_ghost].shape))
    tile = tile.at[lower_ghost].set(0.0)
    return tile.at[upper_ghost].set(0.0)


def _conducting_axis(tile, axis, g, axis_name, axis_size, positive, negative,
                     location, parity, fold):
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
                a = a.at[_plane(axis, g+owner)].add(sign*original[_plane(axis, i)])
                a = a.at[_plane(axis, i)].set(0.)
            else:
                a = a.at[_plane(axis, i)].set(sign*original[_plane(axis, g+owner)])
        return a
    endpoint_plane = _plane(axis, g+n)
    if fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
        if not staggered:
            tile = jax.lax.cond(index == 0,
                lambda a: a.at[_plane(axis, g)].multiply(1+parity), lambda a: a, tile)
            tile = jax.lax.cond(index == axis_size-1,
                lambda a: a.at[endpoint_plane].multiply(1+parity), lambda a: a, tile)
        endpoint = tile[endpoint_plane]
        # Reflection has already folded the exterior sources. Exchange only
        # inter-tile deposits, using the ordinary nonperiodic communication.
        tile = _exchange_axis(tile, axis, g, axis_name, axis_size, BC_ABSORBING,
                              positive, negative, True)
    else:
        endpoint = tile[endpoint_plane]
        tile = _exchange_axis(tile, axis, g, axis_name, axis_size, BC_CONDUCTING,
                              positive, negative, False)
    if not staggered:
        tile = jax.lax.cond(index == axis_size-1,
                            lambda a: a.at[endpoint_plane].set(endpoint), lambda a: a, tile)
    if not fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
    return tile


def _wall_axis(tile, axis, g, axis_name, axis_size, constant):
    """Write the global wall of one axis on the two tiles that own it.

    ``constant`` fills the exterior ghost slab from the adjacent interior
    plane; otherwise the physical endpoint plane is zeroed.
    """

    index = jax.lax.axis_index(axis_name)
    ends = (
        (0, slice(0, g), slice(g, g + 1), g),
        (axis_size - 1, slice(-g, None), slice(-g - 1, -g), -g),
    )
    for owner, ghost, interior, endpoint in ends:
        target = _plane(axis, ghost if constant else endpoint)
        source = _plane(axis, interior)

        def write(a, target=target, source=source):
            if not constant:
                return a.at[target].set(0.0)
            return a.at[target].set(jnp.broadcast_to(a[source], a[target].shape))

        tile = jax.lax.cond(index == owner, write, lambda a: a, tile)
    return tile


def _sweep_tile(tile, g, boundary_conditions, reduced_axes, mesh_shape,
                send_positive, send_negative, *, fold=False,
                reflecting_parity=None, location=None, preserve_exterior=False):
    """Refresh or fold every axis of one tile in x -> y -> z order."""

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
        axis_size = mesh_shape[axis]
        if (boundary_condition == BC_CONDUCTING and location is not None
                and (fold or parity is not None)):
            tile = _conducting_axis(tile, axis, g, axis_name, axis_size,
                                    positive, negative, location[axis], parity, fold)
            continue
        keep_exterior = (preserve_exterior[axis] if isinstance(preserve_exterior, tuple)
                         else preserve_exterior)
        if not fold and (keep_exterior and boundary_condition != BC_PERIODIC or
                         boundary_condition == BC_CONDUCTING and location is not None):
            # C endpoints are owned by the final tile. Restore them before
            # transverse exchange, along with computed exterior values when
            # requested by the metric constitutive/projector operators.
            lower, upper, _, _ = _halo_slices(axis, g)
            before = tile
            tile = _exchange_axis(tile, axis, g, axis_name, axis_size,
                                  boundary_condition, positive, negative, False)
            index = jax.lax.axis_index(axis_name)
            if keep_exterior:
                tile = jax.lax.cond(index == 0,
                    lambda a: a.at[lower].set(before[lower]), lambda a: a, tile)
                tile = jax.lax.cond(index == axis_size-1,
                    lambda a: a.at[upper].set(before[upper]), lambda a: a, tile)
            elif location[axis] == 'C':
                face = _plane(axis, -g)
                tile = jax.lax.cond(index == axis_size-1,
                    lambda a: a.at[face].set(before[face]), lambda a: a, tile)
            continue
        # A normal staggered source owns the lower absorbing face even though
        # it is stored in a halo. Preserve it before the transverse directions
        # are processed, so corner sources receive their transverse folding.
        keep_face = (boundary_condition == BC_ABSORBING and location is not None
                     and parity is not None and location[axis] == 'V')
        if keep_face:
            face = _plane(axis, g-1)
            lower_flux = tile[face]
        if reduced_axis:
            tile = _reduced_axis(tile, axis, g, boundary_condition, fold)
        else:
            tile = _exchange_axis(tile, axis, g, axis_name, axis_size,
                                  boundary_condition, positive, negative, fold)
            if not fold and boundary_condition == BC_CONSTANT:
                tile = _wall_axis(tile, axis, g, axis_name, axis_size, True)
        if keep_face:
            tile = jax.lax.cond(jax.lax.axis_index(axis_name) == 0,
                lambda a: a.at[face].set(lower_flux), lambda a: a, tile)

    return tile


# ---------------------------------------------------------------------------
# shard_map plumbing
# ---------------------------------------------------------------------------

def _is_stacked(field_tiles):
    """True for a vector field held as one stacked seven-dimensional array."""
    return getattr(field_tiles, "ndim", 0) == 7


def _is_vector(field_tiles):
    """True for a stacked or sequence-of-components vector layout."""
    return isinstance(field_tiles, (tuple, list)) or _is_stacked(field_tiles)


def _shard_tile_op(kernel, mesh, vector):
    """Wrap a tile-local kernel in ``shard_map`` over the field mesh.

    The returned callable validates that the leading tile topology matches the
    device mesh and, for vector fields, stacks a component sequence for the
    exchange and restores the caller's layout afterwards.
    """

    mesh_shape = tuple(int(width) for width in mesh.devices.shape)

    def local_op(local_tiles):
        def one_component(local_component):
            tile = kernel(local_component[0, 0, 0])
            return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

        if vector:
            return jax.vmap(one_component, in_axes=0, out_axes=0)(local_tiles)
        return one_component(local_tiles)

    spec = VECTOR_TILE_SPEC if vector else SCALAR_TILE_SPEC
    mapped_op = jax.shard_map(local_op, mesh=mesh, in_specs=spec, out_specs=spec,
                              check_vma=False)

    def run(field_tiles):
        stacked = field_tiles
        if vector and not _is_stacked(field_tiles):
            stacked = jnp.stack(field_tiles, axis=0)
        leading = stacked.shape[1:4] if vector else stacked.shape[:3]
        if tuple(int(width) for width in leading) != mesh_shape:
            kind = "vector" if vector else "field"
            raise ValueError(
                f"Tiled {kind} communication requires one logical tile per device: "
                f"field tile topology {tuple(int(width) for width in leading)} "
                f"does not match device mesh {mesh_shape}."
            )
        result = mapped_op(stacked)
        if vector and not _is_stacked(field_tiles):
            return tuple(result[i] for i in range(int(result.shape[0])))
        return result

    return run


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def make_tiled_halo_op(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    fold=False,
    vector=False,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    """
    Build a shard-mapped halo refresher (``fold=False``) or deposit folder.

    ``vector`` selects the component-batched exchange, which preserves a
    stacked or sequence layout and keeps the component axis general so related
    spatial operators can batch several derivative channels through the same
    communication. ``reflecting_parity`` and ``location`` drive the nodal
    method of images at conducting walls; ``preserve_exterior`` keeps computed
    nonperiodic exterior slabs on refresh (a bool, or an x/y/z tuple of bools).

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned callable instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(int(value) for value in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = tuple(width == 1 and count == 1
                         for width, count in zip(tile_shape, mesh_shape))
    permutations = tuple(axis_permutations(axis_size, bc)
                         for axis_size, bc in zip(mesh_shape, boundary_conditions))
    send_positive = tuple(pair[0] for pair in permutations)
    send_negative = tuple(pair[1] for pair in permutations)

    kernel = partial(
        _sweep_tile,
        g=g,
        boundary_conditions=boundary_conditions,
        reduced_axes=reduced_axes,
        mesh_shape=mesh_shape,
        send_positive=send_positive,
        send_negative=send_negative,
        fold=fold,
        reflecting_parity=reflecting_parity,
        location=location,
        preserve_exterior=preserve_exterior,
    )
    return _shard_tile_op(kernel, mesh, vector)


def make_tiled_wall_op(mesh, axis, num_guard_cells, *, constant=False):
    """
    Build a shard-mapped global-wall writer for one spatial axis.

    ``constant=False`` zeros the physical endpoint planes; ``constant=True``
    fills the exterior ghost cells from the adjacent interior plane.
    """

    g = int(num_guard_cells)
    axis = int(axis)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    kernel = partial(_wall_axis, axis=axis, g=g, axis_name=MESH_AXES[axis],
                     axis_size=mesh_shape[axis], constant=bool(constant))
    return _shard_tile_op(kernel, mesh, False)


# ---------------------------------------------------------------------------
# Field and particle boundary adapters
# ---------------------------------------------------------------------------

def _static_boundaries(static_parameters, bc_type):
    """Per-axis field boundary codes as plain Python integers."""

    if int(bc_type) != BC_TYPE_FIELD:
        raise ValueError("bc_type must be 0 for field boundaries or 1 for particle boundaries.")
    boundary_conditions = static_parameters.boundary_conditions
    if not isinstance(boundary_conditions, tuple):
        boundary_conditions = tuple(boundary_conditions[key] for key in ("x", "y", "z"))
    return tuple(int(jax.device_get(value)) for value in boundary_conditions)


def particle_vector_reflecting_parity(component):
    """Return specular-reflection parity for a deposited Cartesian component.

    The normal component is odd and the two tangential components are even.
    The returned tuple is ordered by wall normal as ``(x, y, z)``.
    """

    return tuple(-1 if axis == int(component) else 1 for axis in range(3))


@partial(jax.jit, static_argnames=('static', 'location', 'parity', 'fold', 'particle'))
def _scalar_source_boundaries(value, static, location=('C', 'C', 'C'), parity=(1, 1, 1),
                              *, fold=False, particle=False):
    """Fold deposits or refresh values with an explicit location and parity."""
    bc = static.particle_boundary_conditions if particle else static.boundary_conditions
    return make_tiled_halo_op(static.field_mesh, static.tile_shape, bc, static.guard_cells,
                              fold=fold, reflecting_parity=parity, location=location)(value)


def source_boundaries(value, static, *, fold=False, location=None,
                      reflecting_parity=None, particle=True):
    """Use charge-node or current-component ownership at deposited boundaries.

    A scalar value folds or refreshes at ``location`` (collocated C nodes by
    default) with ``reflecting_parity``, even on every axis by default. A
    vector value uses the Yee E-component locations, with odd normal and even
    tangential parity per component. ``particle=False`` reads the field
    boundary codes instead of the particle ones.
    """

    if not _is_vector(value):
        return _scalar_source_boundaries(
            value, static,
            ('C', 'C', 'C') if location is None else location,
            (1, 1, 1) if reflecting_parity is None else reflecting_parity,
            fold=fold, particle=particle,
        )
    locations = D_FIELD_LOCATIONS if location is None else location
    result = []
    for component, (array, component_location) in enumerate(zip(value, locations)):
        parity = (
            particle_vector_reflecting_parity(component)
            if reflecting_parity is None else reflecting_parity[component]
        )
        result.append(_scalar_source_boundaries(
            array, static, component_location, parity, fold=fold, particle=particle
        ))
    return jnp.stack(result) if _is_stacked(value) else tuple(result)


def _exchange_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells, bc_type,
                                fold, reflecting_parity, location, preserve_exterior):
    """Shared body of the public refresh and fold adapters."""

    vector = _is_vector(field_tiles)
    if bc_type == BC_TYPE_PARTICLE:
        return source_boundaries(
            field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
            fold=fold, location=location, reflecting_parity=reflecting_parity)

    if reflecting_parity is not None:
        # Reflection parity describes particle deposits; field walls ignore it.
        raise ValueError("reflecting_parity is only valid for particle boundary conditions.")

    boundary_conditions = _static_boundaries(static_parameters, bc_type)
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    preserve_any = (any(preserve_exterior) if isinstance(preserve_exterior, tuple)
                    else preserve_exterior)
    if vector and not fold and (preserve_any or location is not None
                                and BC_CONDUCTING in boundary_conditions):
        # Conducting walls with staggered locations, or preserved exterior
        # slabs, refresh each component at its own location.
        locations = (None,) * len(field_tiles) if location is None else location
        result = tuple(
            make_tiled_halo_op(
                static_parameters.field_mesh, tile_shape, boundary_conditions,
                num_guard_cells, location=component_location,
                preserve_exterior=preserve_exterior)(value)
            for value, component_location in zip(field_tiles, locations))
        return jnp.stack(result) if _is_stacked(field_tiles) else result

    return make_tiled_halo_op(
        static_parameters.field_mesh, tile_shape, boundary_conditions, num_guard_cells,
        fold=fold, vector=vector,
        location=None if vector else location,
        preserve_exterior=False if vector else preserve_exterior,
    )(field_tiles)


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
    Refresh tile halos of a scalar or vector tiled field.

    The layout is read from ``field_tiles``: a six-dimensional array is a
    scalar, while a stacked seven-dimensional array or a sequence of
    components is a vector whose layout is preserved. ``location`` is one
    ``(x, y, z)`` triple of ``'C'``/``'V'`` for a scalar and one triple per
    component for a vector; it preserves owned upper C endpoints at conducting
    walls. Particle boundaries mirror exterior halos using
    ``reflecting_parity``; scalar particle deposits are even by default and
    particle vectors default to odd normal and even tangential parity.
    ``preserve_exterior`` keeps computed nonperiodic exterior slabs (a bool, or
    an x/y/z tuple of bools).
    """

    return _exchange_tiled_ghost_cells(
        field_tiles, static_parameters, num_guard_cells, bc_type, False,
        reflecting_parity, location, preserve_exterior)


def fold_tiled_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
    location=None,
):
    """
    Add tile-ghost deposits into owning interiors, then clear ghost cells.

    Folding uses the same x -> y -> z order as halo refresh and accepts the
    same scalar and vector layouts. A ghost deposit is sent back to the
    neighboring interior that owns it; conducting exterior deposits reflect
    only on devices touching the true global walls. Particle deposits use the
    nodal method of images about the wall nodes, the same for every solver:
    charge sits on collocated C nodes and currents on the Yee E-component
    locations, odd in the normal component and even in the two tangential
    ones.
    """

    return _exchange_tiled_ghost_cells(
        field_tiles, static_parameters, num_guard_cells, bc_type, True,
        reflecting_parity, location, False)


def apply_tiled_zero_boundary(field_tiles, static_parameters, axis, num_guard_cells=2, *, location=None):
    """
    Zero scalar values on the global conducting wall for one spatial axis.
    """

    axis = int(axis)
    boundary_conditions = _static_boundaries(static_parameters, BC_TYPE_FIELD)
    if boundary_conditions[axis] == BC_CONDUCTING:
        apply_bc = make_tiled_wall_op(static_parameters.field_mesh, axis, num_guard_cells)
        field_tiles = apply_bc(field_tiles)
    return update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells,
                                    location=location)


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

    boundary_conditions = _static_boundaries(static_parameters, BC_TYPE_FIELD)
    if boundary_conditions[axis] not in (BC_CONDUCTING, BC_CONSTANT):
        return field_tiles

    apply_bc = make_tiled_wall_op(static_parameters.field_mesh, axis, num_guard_cells,
                                  constant=True)
    return apply_bc(field_tiles)
