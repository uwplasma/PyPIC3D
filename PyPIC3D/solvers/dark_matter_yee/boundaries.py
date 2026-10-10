"""Homogeneous Proca conductor walls at the actual Yee endpoints."""

from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells, update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.halo_exchange import make_distributed_ghost_updater
from PyPIC3D.boundary_conditions.ownership import face_mask
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS


@partial(jax.jit, static_argnames=("static", "kind"))
def dark_field_boundaries(field, static, kind="E"):
    """Refresh phi, electric-edge E/A, or magnetic-face B, including PEC images.

    phi and tangential E/A are odd; normal E/A are even. B has the opposite
    vector parity. Only odd C nodes lie on a wall and must be clamped to zero.
    The scalar distributed primitive supplies global-wall and corner images;
    particle-deposition reflection rules and other field solvers are untouched.
    """
    scalar = kind == "phi"
    if scalar:
        locations = (("C", "C", "C"),)
    elif kind == "B":
        locations = B_FIELD_LOCATIONS
    else:
        locations = D_FIELD_LOCATIONS
    if BC_CONDUCTING not in static.boundary_conditions:
        if scalar:
            return update_tiled_ghost_cells(field, static, static.guard_cells, location=locations[0])
        return update_tiled_vector_ghost_cells(field, static, static.guard_cells, locations=locations)

    result = []
    for component, (value, location) in enumerate(zip((field,) if scalar else field, locations)):
        if scalar:
            parity = (-1, -1, -1)
        elif kind == "B":
            parity = tuple(-1 if axis == component else 1 for axis in range(3))
        else:
            parity = tuple(1 if axis == component else -1 for axis in range(3))
        for axis, bc in enumerate(static.boundary_conditions):
            if bc == BC_CONDUCTING and location[axis] == "C" and parity[axis] == -1:
                wall = face_mask(value.shape, axis, static.tile_shape[axis], static.guard_cells)
                value = jnp.where(wall, 0., value)
        refresh = make_distributed_ghost_updater(
            static.field_mesh, static.tile_shape, static.boundary_conditions, static.guard_cells,
            reflecting_parity=parity, location=location,
        )
        result.append(refresh(value))
    return result[0] if scalar else tuple(result)
