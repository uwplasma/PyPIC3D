"""Exact local relaxation about a fixed native density, with midpoint forcing."""

import math
import jax.numpy as jnp

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS


def make_reference_sponge(displacement, magnetic, static_parameters, dynamic_parameters,
                         *, width, order=4, target_attenuation=1.e-3):
    """Build relaxation rates and fixed exterior references for native D/B.

    Width is a coordinate length. References include native ghost samples;
    they remain fixed densities when geometry evolves. Nonperiodic axes must
    have constant exterior boundaries. The attenuation assumes unit ray speed.
    """
    if width <= 0 or order < 1 or not 0 < target_attenuation < 1:
        raise ValueError("Sponge needs positive width/order and attenuation between zero and one")
    boundaries = static_parameters.boundary_conditions
    if any(bc not in (0, 3) for bc in boundaries):
        raise ValueError("Reference sponge requires periodic or constant exterior axes")
    spacing = (dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)
    counts = displacement[0].shape[:3]
    sizes = tuple(count * tile for count, tile in zip(counts, static_parameters.tile_shape))
    if any(bc and 2 * width >= size * float(dx) for bc, size, dx in zip(boundaries, sizes, spacing)):
        raise ValueError("Opposite sponge layers must leave an undamped interior")
    sigma_max = -(order + 1) * math.log(target_attenuation) / width
    g = static_parameters.guard_cells

    result = []
    for reference, locations in ((displacement, D_FIELD_LOCATIONS), (magnetic, B_FIELD_LOCATIONS)):
        rates, masks = [], []
        for initial, location in zip(reference, locations):
            sigma = jnp.zeros_like(initial)
            exterior = jnp.zeros(initial.shape, dtype=bool)
            for axis, (bc, size, dx) in enumerate(zip(boundaries, sizes, spacing)):
                if bc == 0:
                    continue
                grid = (dynamic_parameters.grids.tiled_vertex_grid if location[axis] == "V"
                        else dynamic_parameters.grids.tiled_center_grid)
                dimensions = counts + tuple(initial.shape[3 + a] if a == axis else 1 for a in range(3))
                coordinate = grid[axis].reshape(dimensions)
                lower = dynamic_parameters.grids.center[axis][1] - dx / 2
                upper = lower + size * dx
                for distance in (coordinate - lower, upper - coordinate):
                    sigma = sigma + sigma_max * jnp.clip(1 - distance / width, 0, 1)**order
                index = (jnp.arange(counts[axis])[:, None] * static_parameters.tile_shape[axis]
                         + jnp.arange(initial.shape[axis + 3])[None, :] - g)
                shape = [1] * 6
                shape[axis], shape[axis + 3] = index.shape
                outside = (index < 0) | (index >= size - (location[axis] == "V"))
                exterior = exterior | outside.reshape(shape)
            rates.append(sigma)
            masks.append(exterior)
        result.append((tuple(rates), tuple(reference), tuple(masks)))
    return tuple(result)


def sponge_advance(value, rhs, dt, sigma, reference):
    """Integrate dF/dt = rhs - sigma*(F-reference), freezing midpoint rhs.

    This exponential midpoint step is second order for smooth forcing. The
    zero-rate branch retains the original Yee update, including its rounding.
    Negative dt is supported for initializing staggered field histories.
    """
    decay = jnp.exp(-sigma * dt)
    weight = -jnp.expm1(-sigma * dt) / jnp.where(sigma != 0, sigma, 1.0)
    relaxed = reference + decay * (value - reference) + weight * rhs
    return jnp.where(sigma != 0, relaxed, value + dt * rhs)
