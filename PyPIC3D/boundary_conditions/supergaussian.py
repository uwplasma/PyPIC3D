import math

import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING


SUPERGAUSSIAN_WALLS = ["-x", "+x", "-y", "+y", "-z", "+z"]

_AXIS_FOR_WALL = {"-x": "x", "+x": "x", "-y": "y", "+y": "y", "-z": "z", "+z": "z"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _active_cells(dynamic_parameters, axis):
    if axis == "x":
        return int(dynamic_parameters.Nx)
    if axis == "y":
        return int(dynamic_parameters.Ny)
    return int(dynamic_parameters.Nz)


def _axis_spacing(dynamic_parameters, axis):
    if axis == "x":
        return float(dynamic_parameters.dx)
    if axis == "y":
        return float(dynamic_parameters.dy)
    return float(dynamic_parameters.dz)


def load_supergaussian_from_toml(raw_supergaussian, dynamic_parameters):
    """
    Read supergaussian absorbing layers from TOML-style config.

    Each layer tuple is `(axis_index, side, width, order, sigma_max)`, where
    side is `-1` for the lower wall and `+1` for the upper wall.  The finished
    tuple is hashable and can live in `StaticParameters` without carrying array
    state through the JAX cache key.
    """
    raw_layers = [] if raw_supergaussian is None else raw_supergaussian
    raw_layers = [raw_layers] if isinstance(raw_layers, dict) else list(raw_layers)

    layers = []
    seen_walls = set()
    for raw in raw_layers:
        wall = raw.get("wall")
        if wall not in SUPERGAUSSIAN_WALLS:
            raise ValueError(f"Invalid supergaussian wall: {wall}. Expected one of {SUPERGAUSSIAN_WALLS}")
        if wall in seen_walls:
            raise ValueError(f"Duplicate supergaussian wall: {wall}")
        seen_walls.add(wall)

        axis = _AXIS_FOR_WALL[wall]
        axis_index = _AXIS_INDEX[axis]
        width = int(raw.get("width", 0))
        active_cells = _active_cells(dynamic_parameters, axis)

        if width <= 0:
            raise ValueError(f"Supergaussian width for {wall} must be positive")
        if width > active_cells:
            raise ValueError(
                f"Supergaussian width for {wall} exceeds active cells on {axis}: {width} > {active_cells}"
            )

        order = float(raw.get("order", 4.0))
        target_reflection = float(raw.get("target_reflection", 1.0e-8))
        if "sigma_max" in raw:
            sigma_max = float(raw["sigma_max"])
        else:
            layer_width = width * _axis_spacing(dynamic_parameters, axis)
            sigma_max = -((order + 1.0) * float(dynamic_parameters.C) * math.log(target_reflection)) / (
                2.0 * layer_width
            )

        side = -1 if wall[0] == "-" else 1
        layers.append((axis_index, side, width, order, sigma_max))

    sg_x = any(axis_index == 0 for axis_index, _, _, _, _ in layers)
    sg_y = any(axis_index == 1 for axis_index, _, _, _, _ in layers)
    sg_z = any(axis_index == 2 for axis_index, _, _, _, _ in layers)

    return bool(layers), sg_x, sg_y, sg_z, tuple(layers)


def _shape_from_static_parameters(static_parameters):
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    g = int(static_parameters.guard_cells)
    mesh_shape = tuple(int(static_parameters.field_mesh.shape[axis]) for axis in ghost_cells.MESH_AXES)
    return (
        mesh_shape[0],
        mesh_shape[1],
        mesh_shape[2],
        tile_nx + 2 * g,
        tile_ny + 2 * g,
        tile_nz + 2 * g,
    )


def _axis_envelope(shape, static_parameters, axis_index, side, width, order, sigma_max, step_dt):
    tile_width = int(static_parameters.tile_shape[axis_index])
    g = int(static_parameters.guard_cells)
    tile_count = int(shape[axis_index])
    local_width = int(shape[axis_index + 3])
    active_cells = tile_count * tile_width

    tile_index = jnp.arange(tile_count)[:, None]
    local_index = jnp.arange(local_width)[None, :] - g
    global_index = tile_index * tile_width + local_index

    if int(side) < 0:
        distance = global_index
    else:
        distance = active_cells - 1 - global_index

    inside_domain = (global_index >= 0) & (global_index < active_cells)
    inside_layer = inside_domain & (distance >= 0) & (distance < int(width))
    eta = (int(width) - distance) / int(width)
    sigma = float(sigma_max) * eta**float(order)
    envelope = jnp.where(inside_layer, jnp.exp(-sigma * step_dt), 1.0)

    broadcast_shape = [1, 1, 1, 1, 1, 1]
    broadcast_shape[axis_index] = tile_count
    broadcast_shape[axis_index + 3] = local_width
    return envelope.reshape(tuple(broadcast_shape))


def build_supergaussian_envelope(static_parameters, dynamic_parameters, step_dt, shape=None):
    """
    Build the tile-local multiplicative field envelope.

    The mask is one outside requested absorbing layers and decreases smoothly
    toward each selected wall.  Reduced dimensions with a single active cell are
    handled by the same tile-local indexing as ordinary 1D/2D Yee fields.
    """
    del dynamic_parameters

    if shape is None:
        shape = _shape_from_static_parameters(static_parameters)

    envelope = jnp.ones(tuple(int(value) for value in shape))
    for axis_index, side, width, order, sigma_max in static_parameters.supergaussian_layers:
        envelope = envelope * _axis_envelope(
            shape,
            static_parameters,
            axis_index,
            side,
            width,
            order,
            sigma_max,
            step_dt,
        )

    return envelope


def apply_tiled_supergaussian_absorber(field_tiles, static_parameters, dynamic_parameters, step_dt, *, locations=None):
    """
    Multiply a tiled vector field by the configured supergaussian envelope.
    """
    if not static_parameters.supergaussian_active:
        return field_tiles

    envelope = build_supergaussian_envelope(
        static_parameters,
        dynamic_parameters,
        step_dt,
        shape=field_tiles[0].shape,
    )
    damped = tuple(component * envelope for component in field_tiles)

    return ghost_cells.update_tiled_vector_ghost_cells(
        damped,
        static_parameters,
        num_guard_cells=int(static_parameters.guard_cells),
        locations=locations,
        preserve_exterior=tuple(bc == BC_CONDUCTING for bc in static_parameters.boundary_conditions) if locations is not None else False,
    )
