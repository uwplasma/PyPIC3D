"""Boundary staging for native D/B densities and computed auxiliary fields."""
from functools import partial

import jax

from .grid_and_stencil import BC_CONDUCTING, BC_CONSTANT
from .ghost_cells import update_tiled_vector_ghost_cells
from .pec import enforce_pec_B, enforce_pec_D
from PyPIC3D.relativity.field_interpolation import copy_densities


def freeze_horizon_layers(vector, static):
    """Copy the first unfrozen radial plane into the inner radial layers.

    This spherical Kerr-Schild policy is validated during configuration and
    must precede conducting-wall projection. Auxiliary E/H do not use it.
    D/B densities apply it through ``copy_densities``, which holds the
    physical field constant across the frozen layers.
    """
    cells, guard = static.horizon_field_cells, static.guard_cells
    if not cells:
        return vector
    if cells >= static.tile_shape[0]:
        raise ValueError('Horizon boundary reference must lie in the first radial tile')
    reference = guard + cells
    return tuple(
        value.at[0, :, :, :reference].set(value[0, :, :, reference:reference+1])
        for value in vector
    )


@partial(jax.jit, static_argnames=('static', 'locations', 'field_kind'))
def refresh_fields(vector, static, locations, field_kind=None, metric=None):
    """Refresh fields; D/B are native densities and require the Yee metric.

    Auxiliary E/H retain their computed physical exterior values. Only D/B
    receive horizon treatment and the FIDO projector boundary contract, which
    act on the physical vector: every copy into another node rescales by the
    ratio of target to source ``sqrt_gamma``.
    """
    if field_kind not in (None, 'D', 'B'):
        raise ValueError("field_kind must be D, B, or None for auxiliary fields")
    if field_kind is None:
        return update_tiled_vector_ghost_cells(
            vector, static, static.guard_cells, locations=locations, preserve_exterior=True)
    if metric is None:
        raise ValueError('D/B density refresh requires a Yee metric')

    volumes = tuple(sample.sqrt_gamma for sample in (metric.D if field_kind == 'D' else metric.B))
    if static.horizon_field_cells:
        vector = copy_densities(partial(freeze_horizon_layers, static=static), vector, volumes)
    if BC_CONDUCTING in static.boundary_conditions:
        enforce_pec = enforce_pec_D if field_kind == 'D' else enforce_pec_B
        return enforce_pec(vector, static, locations, metric)
    exchange = partial(update_tiled_vector_ghost_cells, static_parameters=static,
                       num_guard_cells=static.guard_cells, locations=locations)
    # Periodic and tile-seam halos copy the same physical point; constant
    # extrapolation copies the adjacent physical value.
    if BC_CONSTANT in static.boundary_conditions:
        return copy_densities(exchange, vector, volumes)
    return exchange(vector)
