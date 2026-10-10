"""Local metric projections for coordinate-aligned, stationary FIDO walls.

All vectors are native densities ``sqrt(gamma) V^i``. Each native wall row is
projected from a co-located density reconstruction; the projector rows are
homogeneous, so projecting the density projects the physical vector. Where
two walls meet and off-diagonal gamma couples them, the coupled D edge rows
and B edge ghosts are solved exactly. Every copy or reflection into another
node rescales by ``sqrt_gamma(target)/sqrt_gamma(source)``, so the physical
vector obeys the FIDO wall policy.
"""
from functools import partial
from itertools import combinations

import jax.numpy as jnp

from .grid_and_stencil import BC_CONDUCTING, BC_CONSTANT
from .ghost_cells import update_tiled_vector_ghost_cells
from .ownership import boundary_plane, face_mask, staggered_mirror
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import copy_densities, metric_at_location, reconstruct_vector


def _normal_component(inverse, vector, axis, component):
    """One row of the coordinate-face projector G applied to a vector."""
    return inverse[..., component, axis] / inverse[..., axis, axis] * vector[axis]


def _reflect_vector(vector, metric, location, axis, static, field_kind):
    """Reflect a co-located density about one pair of physical faces.

    The reflection acts on the physical vector, so each ghost is the
    reflected owner density times ``sqrt_gamma(ghost)/sqrt_gamma(owner)``.
    """
    shape = vector[0].shape
    g, n = static.guard_cells, static.tile_shape[axis]
    total = n*shape[axis]
    surface = list(location)
    surface[axis] = 'C'
    inverse = metric_at_location(metric, tuple(surface)).gamma_inv
    volume = metric_at_location(metric, location).sqrt_gamma
    result = list(vector)
    for low in (True, False):
        tile = 0 if low else shape[axis]-1
        offset = tile*n
        last = g+n-1 if location[axis] == 'V' else g+n
        outside = range(g) if low else range(last+1, shape[axis+3])
        plane = partial(boundary_plane, axis, tile)
        for node in outside:
            owner, walls = staggered_mirror(node-g+offset, total, location[axis])
            components = [v[plane(g+owner-offset)] for v in vector]
            for wall in reversed(walls):
                wall_inverse = inverse[plane(g+wall-offset)]
                normal = [_normal_component(wall_inverse, components, axis, j) for j in range(3)]
                components = [2*normal[j]-components[j] if field_kind == 'D'
                              else components[j]-2*normal[j] for j in range(3)]
            scale = volume[plane(node)] / volume[plane(g+owner-offset)]
            for j in range(3):
                result[j] = result[j].at[plane(node)].set(scale*components[j])
    return tuple(result)


def _exchange_preserving_walls(fields, static, locations, rounds):
    """Communicate owner values without replacing conducting exterior slabs."""
    for _ in range(rounds):
        fields = update_tiled_vector_ghost_cells(
            fields, static, static.guard_cells, locations=locations,
            preserve_exterior=tuple(bc == BC_CONDUCTING for bc in static.boundary_conditions),
        )
    return fields


def _volumes(metric, locations):
    return tuple(metric_at_location(metric, location).sqrt_gamma for location in locations)


def _prepare_normal_D(snapshot, static, axes):
    """Supply the identity normal reflection row before reconstructing wall D.

    This makes the wall trace depend on current owners rather than stale
    exterior input. All faces read the same exchanged snapshot. The caller
    applies it through ``copy_densities`` so each ghost carries its own volume.
    """
    shape = snapshot[0].shape
    guard = static.guard_cells
    prepared = list(snapshot)
    for axis in axes:
        width = static.tile_shape[axis]
        for low in (True, False):
            tile = 0 if low else shape[axis] - 1
            offset = tile * width
            outside = range(guard) if low else range(guard + width, shape[axis + 3])
            plane = partial(boundary_plane, axis, tile)
            for node in outside:
                owner, _ = staggered_mirror(node - guard + offset, width * shape[axis], 'V')
                prepared[axis] = prepared[axis].at[plane(node)].set(
                    snapshot[axis][plane(guard + owner - offset)]
                )
    return tuple(prepared)


def _project_native_nodes(snapshot, static, locations, field_kind, metric, axes):
    """Project faces first, then zero D at intersections of two walls.

    Each B component has exactly one C axis, so only D components (two C
    axes) reach an intersection.
    """
    shape = snapshot[0].shape
    projected = []
    for component, location in enumerate(locations):
        value = snapshot[component]
        at_target = reconstruct_vector(snapshot, locations, location)
        inverse = metric_at_location(metric, location).gamma_inv
        incident = tuple(axis for axis in axes if location[axis] == 'C')
        for count in range(1, len(incident) + 1):
            for faces in combinations(incident, count):
                mask = True
                for axis in faces:
                    mask = mask & face_mask(shape, axis, static.tile_shape[axis], static.guard_cells)
                if count == 1:
                    normal = _normal_component(inverse, at_target, faces[0], component)
                    constrained = normal if field_kind == 'D' else at_target[component] - normal
                else:
                    constrained = 0.
                value = jnp.where(mask, constrained, value)
        projected.append(value)
    return tuple(projected)


def _edge_nodes(shape, static, axes):
    """Yield the corner-tile indices beside every edge where two walls meet.

    For each pair of conducting axes p < q and each of the four corners,
    ``at(p_node, q_node)`` indexes that corner tile with the third axis left
    whole. Each node triple is the wall C node, the owned V node beside it,
    and that V node's mirror ghost.
    """
    g = static.guard_cells
    for p, q in combinations(axes, 2):
        n_p, n_q = static.tile_shape[p], static.tile_shape[q]
        for low_p in (True, False):
            for low_q in (True, False):
                def at(p_node, q_node, p=p, q=q, low_p=low_p, low_q=low_q):
                    index = [slice(None)] * 6
                    index[p], index[p + 3] = (0 if low_p else shape[p] - 1), p_node
                    index[q], index[q + 3] = (0 if low_q else shape[q] - 1), q_node
                    return tuple(index)
                p_nodes = (g, g, g - 1) if low_p else (g + n_p, g + n_p - 1, g + n_p)
                q_nodes = (g, g, g - 1) if low_q else (g + n_q, g + n_q - 1, g + n_q)
                yield p, q, at, p_nodes, q_nodes


def solve_D_edges(projected, snapshot, static, metric, axes):
    """Solve the coupled D wall rows beside each edge where two walls meet.

    On the p-wall, a = D^q at the half-node next to the q-wall reconstructs
    D^p from b = D^p on the q-wall (and b's mirrored normal ghost), and b's
    q-wall row reads a back. One pass over a common snapshot only takes one
    Jacobi step, so both rows are solved together:

        a = a1 + alpha (b - b0),    b = b1 + beta (a - a0)

    where a1, b1 are the one-pass rows computed from snapshot values a0, b0.
    In densities, a's reconstruction averages four D^p nodes with weight 1/4,
    two of which are b and its mirror ghost b_m = b sqrt_gamma(b_m)/sqrt_gamma(b),
    so alpha is the normal-row ratio times (1 + sqrt_gamma(b_m)/sqrt_gamma(b))/4.
    """
    shape = snapshot[0].shape
    locations = D_FIELD_LOCATIONS
    result = list(projected)
    for p, q, at, (p_wall, p_half, p_ghost), (q_wall, q_half, q_ghost) in _edge_nodes(shape, static, axes):
        metric_a = metric_at_location(metric, locations[q])
        metric_b = metric_at_location(metric, locations[p])
        # a third conducting wall turns these nodes into intersections, which stay zero
        on_third_wall = jnp.zeros(shape, bool)
        for r in axes:
            if r not in (p, q):
                on_third_wall = on_third_wall | face_mask(shape, r, static.tile_shape[r], static.guard_cells)

        a, a_mirror = at(p_wall, q_half), at(p_wall, q_ghost)
        b, b_mirror = at(p_half, q_wall), at(p_ghost, q_wall)
        inverse_a = metric_a.gamma_inv[a]
        inverse_b = metric_b.gamma_inv[b]
        # normal-row coefficient times the 2x2 density transfer weight of the pair
        alpha = (inverse_a[..., q, p] / inverse_a[..., p, p]
                 * (metric_b.sqrt_gamma[b] + metric_b.sqrt_gamma[b_mirror])
                 / (4.0 * metric_b.sqrt_gamma[b]))
        beta = (inverse_b[..., p, q] / inverse_b[..., q, q]
                * (metric_a.sqrt_gamma[a] + metric_a.sqrt_gamma[a_mirror])
                / (4.0 * metric_a.sqrt_gamma[a]))

        a0, b0 = snapshot[q][a], snapshot[p][b]
        a1, b1 = projected[q][a], projected[p][b]
        a_solved = (a1 + alpha * (b1 - b0) - alpha * beta * a0) / (1.0 - alpha * beta)
        b_solved = b1 + beta * (a_solved - a0)
        result[q] = result[q].at[a].set(jnp.where(on_third_wall[a], a1, a_solved))
        result[p] = result[p].at[b].set(jnp.where(on_third_wall[b], b1, b_solved))
    return tuple(result)


def solve_B_edges(first, second, static, metric, axes):
    """Solve the coupled B ghost pair beside each edge where two walls meet.

    c = B^p on the p-wall in the first q-ghost and d = B^q on the q-wall in
    the first p-ghost reflect owners that are wall normals (zero), so each is
    only the tangential part of a metric reflection, -2 gamma^pq/gamma^qq,
    of a reconstruction that contains the other. d is reflected in the p pass
    and c in the later q pass, so the second sweep gives

        d2 = beta c1 + d_rest,    c2 = alpha d2 + c_rest

    with rests already exact after the first sweep. Solving the pair gives
    c = c2 + alpha beta Delta and d = d2 + beta Delta, Delta = (c2-c1)/(1-alpha beta).
    In densities, d enters c's owner reconstruction with weight 1/4 and the
    reflection carries sqrt_gamma(c)/sqrt_gamma(c owner) into the ghost.
    """
    shape = first[0].shape
    locations = B_FIELD_LOCATIONS
    result = list(second)
    for p, q, at, (p_wall, p_half, p_ghost), (q_wall, q_half, q_ghost) in _edge_nodes(shape, static, axes):
        c, c_owner = at(p_wall, q_ghost), at(p_wall, q_half)
        d, d_owner = at(p_ghost, q_wall), at(p_half, q_wall)
        # both reflections use the wall metric at the edge node, as in _reflect_vector
        surface = ['V', 'V', 'V']
        surface[p] = surface[q] = 'C'
        inverse = metric_at_location(metric, tuple(surface)).gamma_inv[at(p_wall, q_wall)]
        sqrt_p = metric_at_location(metric, locations[p]).sqrt_gamma
        sqrt_q = metric_at_location(metric, locations[q]).sqrt_gamma
        alpha = -2.0 * inverse[..., p, q] / inverse[..., q, q] * sqrt_p[c] / (4.0 * sqrt_p[c_owner])
        beta = -2.0 * inverse[..., q, p] / inverse[..., p, p] * sqrt_q[d] / (4.0 * sqrt_q[d_owner])

        c1, c2, d2 = first[p][c], second[p][c], second[q][d]
        delta = (c2 - c1) / (1.0 - alpha * beta)
        result[p] = result[p].at[c].set(c2 + alpha * beta * delta)
        result[q] = result[q].at[d].set(d2 + beta * delta)
    return tuple(result)


def _reflect_exterior_axis(snapshot, static, locations, field_kind, metric, axis, previous_axes):
    """Reflect one exterior axis, composing earlier reflections at corners."""
    result = []
    for component, location in enumerate(locations):
        reconstructed = reconstruct_vector(snapshot, locations, location)
        # Native transfers at corners can leave the allocated halo. Reconstruct
        # at the reflected owner first, then compose co-located reflections.
        for previous in previous_axes:
            reconstructed = _reflect_vector(reconstructed, metric, location, previous, static, field_kind)
        reflected = _reflect_vector(reconstructed, metric, location, axis, static, field_kind)
        result.append(reflected[component])
    return tuple(result)


def _pec_setup(shape, static, locations, metric):
    """Conducting axes and the wall-preserving halo exchange for one field.

    Periodic and tile-seam copies join the same physical point. Constant
    extrapolation copies the adjacent physical value, so it rescales volumes.
    """
    axes = tuple(axis for axis, bc in enumerate(static.boundary_conditions) if bc == BC_CONDUCTING)
    guard = static.guard_cells
    # Narrow distributed axes need multiple neighbor passes; reduced periodic
    # directions are filled in one pass.
    rounds = max([1] + [(guard + width - 1) // width
                       for axis, width in enumerate(static.tile_shape) if shape[axis] > 1])
    exchange = partial(_exchange_preserving_walls, static=static, locations=locations, rounds=rounds)
    if BC_CONSTANT in static.boundary_conditions:
        exchange = partial(copy_densities, exchange, volumes=_volumes(metric, locations))
    return axes, exchange


def enforce_pec_D(vector, static, locations, metric):
    """Project D onto conducting walls: zero tangential FIDO field.

    Native wall rows keep only the metric normal part, intersections of
    independent normals are zero, and the rows coupled across each edge are
    solved together. Exterior ghosts are then reflected in x/y/z order.
    Internal boundaries are communicated, never projected.
    """
    axes, exchange = _pec_setup(vector[0].shape, static, locations, metric)

    snapshot = exchange(vector)
    prepare = partial(_prepare_normal_D, static=static, axes=axes)
    snapshot = exchange(copy_densities(prepare, snapshot, _volumes(metric, locations)))
    projected = _project_native_nodes(snapshot, static, locations, 'D', metric, axes)
    projected = solve_D_edges(projected, snapshot, static, metric, axes)
    result = exchange(projected)

    for position, axis in enumerate(axes):
        result = exchange(_reflect_exterior_axis(result, static, locations, 'D', metric, axis, axes[:position]))
    return result


def enforce_pec_B(vector, static, locations, metric):
    """Project B onto conducting walls: zero normal flux.

    Each B component has native nodes on only one wall family, where it is
    the normal component and is set to zero; no B component lies on a wall
    intersection. Exterior ghosts are reflected in x/y/z order; where two
    walls meet, the coupled ghost pair is solved from two reflection sweeps.
    Internal boundaries are communicated, never projected.
    """
    axes, exchange = _pec_setup(vector[0].shape, static, locations, metric)

    def sweep(fields):
        for position, axis in enumerate(axes):
            fields = exchange(_reflect_exterior_axis(fields, static, locations, 'B', metric, axis, axes[:position]))
        return fields

    snapshot = exchange(vector)
    result = exchange(_project_native_nodes(snapshot, static, locations, 'B', metric, axes))
    result = sweep(result)
    if len(axes) > 1:
        second = sweep(result)
        result = exchange(solve_B_edges(result, second, static, metric, axes))
    if len(axes) > 2:
        # triple-corner ghosts read the solved edge pairs; at the fixed point
        # this sweep reproduces the pairs themselves
        result = sweep(result)
    return result
