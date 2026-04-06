import jax
from jax import jit
import jax.numpy as jnp


BC_PERIODIC = 0
BC_CONDUCTING = 1


@jit
def wrap_periodic_position(x, wind):
    """Wrap positions into the principal periodic domain ``[-wind/2, wind/2]``."""
    half_wind = 0.5 * wind
    wrapped = jnp.mod(x + half_wind, wind) - half_wind
    return jnp.where((wrapped == -half_wind) & (x >= half_wind), half_wind, wrapped)


def axis_has_active_cells(axis_size, ghost_cells=False):
    """Return True when an axis represents more than one physical cell."""
    return axis_size > (3 if ghost_cells else 1)


def inactive_axis_index(axis_size, ghost_cells=False):
    """Return the physical singleton-cell index for a reduced-dimension axis."""
    return 1 if ghost_cells and axis_size >= 3 else axis_size // 2


def uniform_axis_spacing(axis):
    """Return the spacing of a uniformly sampled 1D axis."""
    return axis[1] - axis[0] if len(axis) > 1 else 1.0


def compute_particle_anchor(position, grid_axis, shape_factor):
    """Compute the raw stencil anchor index for a particle position."""
    spacing = uniform_axis_spacing(grid_axis)
    origin = grid_axis[0]
    return jax.lax.cond(
        shape_factor == 1,
        lambda _: jnp.floor((position - origin) / spacing).astype(int),
        lambda _: jnp.round((position - origin) / spacing).astype(int),
        operand=None,
    )


def particle_axis_offset(position, anchor, grid_axis):
    """Compute the offset from the raw anchor point used to evaluate weights."""
    spacing = uniform_axis_spacing(grid_axis)
    return position - (anchor * spacing + grid_axis[0])


def build_axis_stencil_points(anchor, axis_size, bc, offsets):
    """Build a particle stencil from raw anchors, wrapping periodic indices when requested."""
    stencil = anchor[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
    return jax.lax.cond(
        bc == BC_PERIODIC,
        lambda pts: jnp.mod(pts, axis_size),
        lambda pts: pts,
        operand=stencil,
    )


def collapse_axis_stencil(points, weights, axis_size, ghost_cells=False):
    """Collapse an inactive axis onto its physical singleton interior cell."""
    if axis_has_active_cells(axis_size, ghost_cells=ghost_cells):
        return points, weights

    collapsed_index = inactive_axis_index(axis_size, ghost_cells=ghost_cells)
    collapsed_points = jnp.full(
        (1, points.shape[1]),
        collapsed_index,
        dtype=points.dtype,
    )
    collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
    return collapsed_points, collapsed_weights


def prepare_particle_axis_stencil(position, grid_axis, axis_size, shape_factor, bc, wind=None, ghost_cells=False):
    """Return raw anchors, raw offsets, and the standard 3-point stencil."""
    del wind
    anchor = compute_particle_anchor(position, grid_axis, shape_factor)
    offset = particle_axis_offset(position, anchor, grid_axis)
    offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)
    points = build_axis_stencil_points(anchor, axis_size, bc, offsets)
    return position, anchor, offset, points


def build_collocated_axis(minimum_physical, spacing, count):
    """Build a collocated axis with one ghost position on each side."""
    start = minimum_physical - spacing
    stop = minimum_physical + count * spacing
    return jnp.linspace(start, stop, count + 2)


def build_staggered_axis(minimum_physical, spacing, count):
    """Build a staggered axis with one ghost position on each side."""
    start = minimum_physical - 0.5 * spacing
    stop = minimum_physical + (count + 0.5) * spacing
    return jnp.linspace(start, stop, count + 2)
