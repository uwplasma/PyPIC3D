# Christopher Woolford April 6th 2026
# Grid construction and particle stencil utilities for ghost-cell-based boundary handling.

import jax
from jax import jit
import jax.numpy as jnp
# import external libraries

# Integer codes that identify the boundary condition type on each axis.
BC_PERIODIC = 0     # periodic (wrap-around); ghost cells mirror opposite interior edge
BC_CONDUCTING = 1   # conducting (Dirichlet); tangential E components vanish at walls


@jit
def wrap_periodic_position(x, wind):
    """
    Wrap particle positions into the principal periodic domain.

    Maps positions into the interval [-wind/2, wind/2] using modular
    arithmetic. The edge case where the wrapped value lands exactly on
    -wind/2 (but the original was at +wind/2) is resolved to +wind/2
    to avoid ambiguity at the domain boundary.

    Args:
        x (jnp.ndarray): Particle positions along one axis (m).
        wind (float): Total physical extent of the periodic domain along this axis (m).

    Returns:
        jnp.ndarray: Wrapped positions in [-wind/2, wind/2].
    """
    half_wind = 0.5 * wind
    wrapped = jnp.mod(x + half_wind, wind) - half_wind
    # Resolve ambiguity: if the unwrapped position was at the +wind/2 edge,
    # keep it there instead of mapping to -wind/2.
    return jnp.where((wrapped == -half_wind) & (x >= half_wind), half_wind, wrapped)


def axis_has_active_cells(axis_size, ghost_cells=False):
    """
    Check whether an axis represents more than one physical cell.

    A "reduced" (inactive) axis has only a single physical cell and is used
    when a simulation dimension is suppressed (e.g., 2D or 1D runs stored
    in a 3D array). With ghost cells the array has 3 entries for a single
    physical cell (ghost | interior | ghost); without ghost cells, 1 entry.

    Args:
        axis_size (int): Total number of grid points along this axis (including ghosts).
        ghost_cells (bool): Whether the axis array includes ghost cell padding.

    Returns:
        bool: True if the axis has more than one physical cell.
    """
    return axis_size > (3 if ghost_cells else 1)


def inactive_axis_index(axis_size, ghost_cells=False):
    """
    Return the grid index of the single physical cell on a reduced-dimension axis.

    When an axis is inactive (single physical cell), all stencil points
    must be collapsed onto this index. With ghost cells the layout is
    [ghost, interior, ghost], so the interior index is 1. Without ghost
    cells the interior index is axis_size // 2 (typically 0).

    Args:
        axis_size (int): Total number of grid points along this axis.
        ghost_cells (bool): Whether the axis array includes ghost cell padding.

    Returns:
        int: Index of the singleton interior cell.
    """
    return 1 if ghost_cells and axis_size >= 3 else axis_size // 2


def uniform_axis_spacing(axis):
    """
    Calculate the uniform spacing of a 1D grid axis.

    Assumes the axis is uniformly sampled. Returns 1.0 for a single-point
    axis to avoid division by zero in downstream calculations.

    Args:
        axis (jnp.ndarray): 1D array of grid coordinates (m).

    Returns:
        float: Grid spacing dx (m).
    """
    return axis[1] - axis[0] if len(axis) > 1 else 1.0


def compute_particle_anchor(position, grid_axis, shape_factor):
    """
    Compute the nearest grid index (anchor) for each particle along one axis.

    For first-order (CIC) shapes, the anchor is the left cell boundary
    (floor). For second-order (TSC) shapes, the anchor is the nearest
    grid node (round). This determines the center of the 3-point stencil
    used for interpolation and deposition.

    Args:
        position (jnp.ndarray): Particle positions along this axis (m).
        grid_axis (jnp.ndarray): 1D grid coordinates for this axis (m).
        shape_factor (int): Particle shape order (1 = CIC, 2 = TSC).

    Returns:
        jnp.ndarray: Integer anchor indices, one per particle.
    """
    spacing = uniform_axis_spacing(grid_axis)
    origin = grid_axis[0]
    # CIC (shape_factor == 1): anchor at the left cell boundary (floor).
    # TSC (shape_factor == 2): anchor at the nearest grid node (round).
    return jax.lax.cond(
        shape_factor == 1,
        lambda _: jnp.floor((position - origin) / spacing).astype(int),
        lambda _: jnp.round((position - origin) / spacing).astype(int),
        operand=None,
    )


def particle_axis_offset(position, anchor, grid_axis):
    """
    Compute the displacement of each particle from its anchor grid node.

    This offset is used to evaluate the particle shape-function weights.
    It is the distance in physical coordinates from the anchor node to
    the particle position.

    Args:
        position (jnp.ndarray): Particle positions along this axis (m).
        anchor (jnp.ndarray): Integer anchor indices from compute_particle_anchor.
        grid_axis (jnp.ndarray): 1D grid coordinates for this axis (m).

    Returns:
        jnp.ndarray: Offset from anchor to particle position (m).
    """
    spacing = uniform_axis_spacing(grid_axis)
    return position - (anchor * spacing + grid_axis[0])


def build_axis_stencil_points(anchor, axis_size, bc, offsets):
    """
    Build stencil grid indices from particle anchors and stencil offsets.

    For periodic BCs, indices that fall outside [0, axis_size) are wrapped
    via modular arithmetic. For conducting BCs, no wrapping is applied
    (out-of-range indices are left as-is for downstream clamping or masking).

    Args:
        anchor (jnp.ndarray): Integer anchor indices, shape (N_particles,).
        axis_size (int): Number of grid points along this axis.
        bc (int): Boundary condition code (BC_PERIODIC or BC_CONDUCTING).
        offsets (jnp.ndarray): Stencil offsets relative to anchor, shape (n_stencil,).

    Returns:
        jnp.ndarray: Stencil indices with shape (n_stencil, N_particles).
    """
    # Broadcasting: offsets[:, newaxis] + anchor[newaxis, :] produces
    # shape (n_stencil, N_particles) -- one column of stencil indices per particle.
    stencil = anchor[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
    return jax.lax.cond(
        bc == BC_PERIODIC,
        lambda pts: jnp.mod(pts, axis_size),
        lambda pts: pts,
        operand=stencil,
    )


def collapse_axis_stencil(points, weights, axis_size, ghost_cells=False):
    """
    Collapse a multi-point stencil to a single point on an inactive axis.

    When an axis has only one physical cell, the 3-point stencil is
    redundant. This function replaces it with a 1-point stencil at the
    singleton interior index, summing all weights so that the total
    contribution is preserved.

    Args:
        points (jnp.ndarray): Stencil indices, shape (n_stencil, N_particles).
        weights (jnp.ndarray): Stencil weights, shape (n_stencil, N_particles).
        axis_size (int): Total number of grid points along this axis.
        ghost_cells (bool): Whether the axis includes ghost cell padding.

    Returns:
        tuple: (collapsed_points, collapsed_weights).
            If the axis is active, the inputs are returned unchanged.
            Otherwise each has shape (1, N_particles).
    """
    if axis_has_active_cells(axis_size, ghost_cells=ghost_cells):
        return points, weights

    # Replace multi-point stencil with a single point at the interior cell.
    # Sum all weights so the total deposited/interpolated value is unchanged.
    collapsed_index = inactive_axis_index(axis_size, ghost_cells=ghost_cells)
    collapsed_points = jnp.full(
        (1, points.shape[1]),
        collapsed_index,
        dtype=points.dtype,
    )
    collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
    return collapsed_points, collapsed_weights


def prepare_particle_axis_stencil(position, grid_axis, axis_size, shape_factor, bc, wind=None, ghost_cells=False):
    """
    Prepare particle stencil data for one axis (anchor, offset, grid indices).

    This is the main entry point for computing the interpolation/deposition
    stencil along a single axis. It computes the anchor index, the
    particle-anchor offset (for shape-function weight evaluation), and
    the wrapped stencil grid indices.

    The ``wind`` parameter is accepted for call-site compatibility with
    earlier code paths but is not used; positions are assumed to already
    lie within the grid domain.

    Args:
        position (jnp.ndarray): Particle positions along this axis (m).
        grid_axis (jnp.ndarray): 1D grid coordinates for this axis (m).
        axis_size (int): Number of grid points along this axis.
        shape_factor (int): Particle shape order (1 = CIC, 2 = TSC).
        bc (int): Boundary condition code (BC_PERIODIC or BC_CONDUCTING).
        wind (float or None): Domain extent (unused; retained for API compatibility).
        ghost_cells (bool): Whether the grid includes ghost cell padding.

    Returns:
        tuple: (position, anchor, offset, stencil_points) where
            - position (jnp.ndarray): The input positions (passed through).
            - anchor (jnp.ndarray): Integer anchor indices.
            - offset (jnp.ndarray): Particle offset from anchor (m).
            - stencil_points (jnp.ndarray): Grid indices for the 3-point stencil,
              shape (3, N_particles).
    """
    del wind  # accepted for call-site compatibility; not used in ghost-cell approach
    anchor = compute_particle_anchor(position, grid_axis, shape_factor)
    offset = particle_axis_offset(position, anchor, grid_axis)
    offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)
    points = build_axis_stencil_points(anchor, axis_size, bc, offsets)
    return position, anchor, offset, points


def build_collocated_axis(minimum_physical, spacing, count):
    """
    Build a 1D collocated grid axis with one ghost cell on each side.

    The physical domain spans [minimum_physical, minimum_physical + count * spacing]
    with ``count`` interior cells. One ghost position is prepended and one appended,
    giving a total of ``count + 2`` points.

    For a collocated grid, nodes sit at cell vertices:
        ghost | phys_0 | phys_1 | ... | phys_{count-1} | ghost

    Args:
        minimum_physical (float): Coordinate of the first physical grid node (m).
        spacing (float): Uniform grid spacing (m).
        count (int): Number of physical grid cells.

    Returns:
        jnp.ndarray: Grid coordinates of length count + 2.
    """
    # Prepend one ghost position at (minimum_physical - spacing) and append one
    # at (minimum_physical + count * spacing).
    start = minimum_physical - spacing
    stop = minimum_physical + count * spacing
    return jnp.linspace(start, stop, count + 2)


def build_staggered_axis(minimum_physical, spacing, count):
    """
    Build a 1D staggered (cell-centered) grid axis with one ghost cell on each side.

    The physical domain is the same as for the collocated grid, but nodes
    are shifted by half a cell so they sit at cell centers:
        ghost | center_0 | center_1 | ... | center_{count-1} | ghost

    Args:
        minimum_physical (float): Coordinate of the first physical cell vertex (m).
        spacing (float): Uniform grid spacing (m).
        count (int): Number of physical grid cells.

    Returns:
        jnp.ndarray: Grid coordinates of length count + 2.
    """
    start = minimum_physical - 0.5 * spacing
    stop = minimum_physical + (count + 0.5) * spacing
    return jnp.linspace(start, stop, count + 2)
