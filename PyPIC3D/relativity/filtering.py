"""Matched periodic Cartesian filters for GR particle--grid coupling.

Sources are filtered as coordinate densities. Particle fields are filtered
after conversion to physical components; the evolved geometry stays intact.
The common symmetric stencil preserves charge continuity, but is not a
discrete variational construction for the nonlinear metric coupling.
"""

from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.relativity.core import Metric
from PyPIC3D.utilities.filters import (
    bilinear_filter, digital_filter, tiled_bilinear_filter, tiled_digital_filter,
)


@partial(jax.jit, static_argnames="boundary_conditions")
def binomial_source_filter(values, boundary_conditions):
    """Filter global coordinate densities, replicating isolated edge samples.

    This source-only operation does not filter native Maxwell fields or their
    charge constraint. Component axes precede the three physical spatial axes.
    """
    padded = values
    for axis, bc in enumerate(boundary_conditions):
        widths = [(0, 0)] * values.ndim
        widths[values.ndim - 3 + axis] = (1, 1)
        padded = jnp.pad(padded, widths, mode="wrap" if bc == 0 else "edge")
    return bilinear_filter(padded)[..., 1:-1, 1:-1, 1:-1]


def validate_gr_filter(static_parameters):
    """The commuting filter requires a periodic Cartesian coordinate grid."""
    if static_parameters.current_filter == "none":
        return
    if static_parameters.current_filter not in ("bilinear", "digital"):
        raise ValueError("GR filtering supports 'none', 'bilinear', or 'digital'")
    if static_parameters.metric not in ("flat_cartesian", "numerical"):
        raise ValueError("Matched GR filtering requires a Cartesian metric")
    if any(bc != 0 for bc in (static_parameters.boundary_conditions
                             + static_parameters.particle_boundary_conditions)):
        raise ValueError("Matched GR filtering requires periodic field and particle boundaries")


@partial(jax.jit, static_argnames=("static_parameters", "tiled"))
def filter_gr_grid(values, static_parameters, dynamic_parameters, *, tiled=False):
    """Apply F to global or tiled grids with optional leading component axes.

    Global arrays have three spatial axes; tiled arrays have three tile axes
    followed by three local spatial axes. Periodic singleton axes are identities
    for the binomial stencil. The digital stencil retains its six-neighbor weights.
    """
    choice = static_parameters.current_filter
    if choice == "none":
        return values
    validate_gr_filter(static_parameters)

    if tiled:
        def filter_component(component):
            if choice == "bilinear":
                return tiled_bilinear_filter(component, static_parameters)
            return tiled_digital_filter(component, dynamic_parameters.alpha, static_parameters)

        if values.ndim == 6:
            return filter_component(values)
        components = values.reshape((-1,) + values.shape[-6:])
        return jax.vmap(filter_component)(components).reshape(values.shape)

    # Reuse the same local stencil as the tiled path. Only the owned samples
    # enter this padding; no duplicate tile ghosts contribute to the source.
    widths = ((0, 0),) * (values.ndim - 3) + ((1, 1),) * 3
    padded = jnp.pad(values, widths, mode="wrap")
    if choice == "bilinear":
        filtered = bilinear_filter(padded)
    else:
        filtered = digital_filter(padded, dynamic_parameters.alpha)
    return filtered[..., 1:-1, 1:-1, 1:-1]


@partial(jax.jit, static_argnames="static_parameters")
def particle_metric(metric, static_parameters, dynamic_parameters):
    """Filter C-node primitives for particles, deriving dependent tensors.

    Hermite sampling subsequently derives particle inverses and gradients from
    these same lapse, shift and covariant-metric interpolants. This center metric
    must never replace the raw Yee metric used by Maxwell or volume factors.
    """
    center = metric.center
    if static_parameters.current_filter == "none":
        return center
    lapse = filter_gr_grid(center.lapse, static_parameters, dynamic_parameters, tiled=True)
    shift = jnp.moveaxis(filter_gr_grid(
        jnp.moveaxis(center.shift, -1, 0), static_parameters, dynamic_parameters, tiled=True), 0, -1)
    gamma = jnp.moveaxis(filter_gr_grid(
        jnp.moveaxis(center.gamma, (-2, -1), (0, 1)),
        static_parameters, dynamic_parameters, tiled=True), (0, 1), (-2, -1))
    return Metric(lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma)))
