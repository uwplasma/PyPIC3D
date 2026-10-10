"""Staggered scalar and deposited-source boundary policies."""
from functools import partial

import jax
import jax.numpy as jnp

from .halo_exchange import (
    make_distributed_ghost_updater, make_distributed_ghost_folder,
    _is_stacked_tiled_vector_field,
)
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS


def particle_vector_reflecting_parity(component):
    """Return specular-reflection parity for a deposited Cartesian component.

    The normal component is odd and the two tangential components are even.
    The returned tuple is ordered by wall normal as ``(x, y, z)``.
    """

    return tuple(-1 if axis == int(component) else 1 for axis in range(3))


@partial(jax.jit, static_argnames=('static', 'location', 'parity', 'fold', 'particle'))
def scalar_boundaries(value, static, location=('C', 'C', 'C'), parity=(1, 1, 1),
                      *, fold=False, particle=False):
    """Fold deposits or refresh values with an explicit location and parity."""
    bc = static.particle_boundary_conditions if particle else static.boundary_conditions
    factory = make_distributed_ghost_folder if fold else make_distributed_ghost_updater
    return factory(static.field_mesh, static.tile_shape, bc, static.guard_cells,
                   reflecting_parity=parity, location=location)(value)


def source_boundaries(value, static, *, fold=False, vector=False, reflecting_parity=None):
    """Use charge-node or current-component ownership at particle boundaries."""
    if not vector:
        parity = (1, 1, 1) if reflecting_parity is None else reflecting_parity
        return scalar_boundaries(value, static, parity=parity, fold=fold, particle=True)
    result = []
    for component, (array, location) in enumerate(zip(value, D_FIELD_LOCATIONS)):
        parity = (
            particle_vector_reflecting_parity(component)
            if reflecting_parity is None else reflecting_parity[component]
        )
        result.append(scalar_boundaries(
            array, static, location, parity, fold=fold, particle=True
        ))
    if _is_stacked_tiled_vector_field(value):
        return jnp.stack(result)
    return tuple(result)
