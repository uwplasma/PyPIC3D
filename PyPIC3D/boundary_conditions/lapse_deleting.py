"""Remove particles inside a prescribed low-lapse region."""

from functools import partial

import jax
from jax.sharding import PartitionSpec as P

from PyPIC3D.relativity.interpolate_metric import (
    interpolate_hermite,
    particle_active_axes,
    safe_inactive_positions,
)


@partial(jax.jit, static_argnames="static_parameters")
def lapse_deleting(particles, metric, static_parameters, dynamic_parameters):
    """Deactivate slots with native interpolated lapse below ``lapse_threshold``.

    This interior condition is independent of the coordinate-domain boundaries.
    Only the activity mask changes; particle storage keeps its fixed JAX shape.
    Charged deletion introduces a charge sink, so endpoint deletion must follow
    the final trajectory-current deposit.
    """
    if not static_parameters.lapse_deleting or not particles.active.size:
        return particles

    g = static_parameters.guard_cells
    active_axes = particle_active_axes(particles.active.shape[:3], static_parameters.tile_shape)

    def one_tile(x_tile, active_tile, lapse, center_grid):
        x_tile, active_tile, lapse, center_grid = jax.tree.map(
            lambda value: value[0, 0, 0], (x_tile, active_tile, lapse, center_grid))
        x = safe_inactive_positions(x_tile, active_tile, center_grid, active_axes, g)
        alpha = interpolate_hermite(lapse, x, center_grid, active_axes, (g, g, g))
        active = active_tile & ~(alpha < static_parameters.lapse_threshold)
        return active[None, None, None]

    # Pass native lapse and coordinates locally: no full metric inversion,
    # coupling filter, or global metric gather is needed for the boundary.
    tile_spec = P("tile_x", "tile_y", "tile_z")
    values = (particles.x, particles.active, metric.center.lapse,
              dynamic_parameters.grids.tiled_center_grid)
    specs = jax.tree.map(lambda value: tile_spec, values)
    active = jax.shard_map(one_tile, mesh=static_parameters.field_mesh,
                          in_specs=specs, out_specs=tile_spec, check_vma=False)(*values)
    return particles._replace(active=active)
