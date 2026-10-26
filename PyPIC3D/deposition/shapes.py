import jax
from jax import jit
import jax.numpy as jnp


def _zeroth_order_axis_weights(delta, spacing):
    """Zeroth-order (NGP) weights along one axis: all weight on the anchor node."""
    del spacing  # the nearest-grid-point shape does not depend on the offset
    zero = jnp.zeros_like(delta)
    return [zero, jnp.ones_like(delta), zero]


def _first_order_axis_weights(delta, spacing):
    """First-order (CIC) weights along one axis."""
    s = delta / spacing
    return [jnp.zeros_like(s), 1 - s, s]


def _second_order_axis_weights(delta, spacing):
    """Second-order (TSC) weights along one axis."""
    s = delta / spacing
    return [(1 / 2) * ((1 / 2) - s) ** 2, (3 / 4) - s ** 2, (1 / 2) * ((1 / 2) + s) ** 2]


@jit
def get_axis_weights(delta, spacing, shape_order):
    """Three-point shape weights along one axis, for shape orders 0, 1 and 2.

    This is the per-axis entry point used by the field gather, where the
    Galerkin ordering gives each axis its own shape order. ``shape_order`` may
    be traced; ``lax.switch`` clamps its index, so an out-of-range order can
    never read past the branch list.

    Args:
        delta: Particle offset from the anchor node along this axis.
        spacing: Grid spacing along this axis.
        shape_order: Particle shape order (0 = NGP, 1 = CIC, 2 = TSC).

    Returns:
        jnp.ndarray: Weights with shape ``(3, N_particles)``.
    """
    return jnp.asarray(jax.lax.switch(
        shape_order,
        [_zeroth_order_axis_weights, _first_order_axis_weights, _second_order_axis_weights],
        delta,
        spacing,
    ))


@jit
def get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """Calculate the second-order weights for particle deposition.

    Args:
        deltax, deltay, deltaz: Particle offsets from the nearest grid point.
        dx, dy, dz: Grid spacings.

    Returns:
        (x_weights, y_weights, z_weights): each a length-3 list of weights.
    """
    return (
        _second_order_axis_weights(deltax, dx),
        _second_order_axis_weights(deltay, dy),
        _second_order_axis_weights(deltaz, dz),
    )


@jit
def get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """Calculate the first-order (CIC) weights for particle deposition."""
    return (
        _first_order_axis_weights(deltax, dx),
        _first_order_axis_weights(deltay, dy),
        _first_order_axis_weights(deltaz, dz),
    )


@jit
def get_zeroth_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """Calculate the zeroth-order (NGP) weights for particle deposition."""
    return (
        _zeroth_order_axis_weights(deltax, dx),
        _zeroth_order_axis_weights(deltay, dy),
        _zeroth_order_axis_weights(deltaz, dz),
    )
