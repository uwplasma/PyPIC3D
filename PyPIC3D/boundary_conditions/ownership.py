"""Owned staggered endpoints and reflected image indices on tiled grids."""
import jax.numpy as jnp

from .grid_and_stencil import BC_CONDUCTING


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

