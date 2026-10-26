from jax import jit
import jax.numpy as jnp
from functools import partial


from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    axis_has_active_cells,
    inactive_axis_index,
    prepare_particle_axis_stencil,
    uniform_axis_spacing,
)


from PyPIC3D.deposition.shapes import get_axis_weights


def galerkin_ordering(location, shape_order):
    """
    Per-axis gather shape orders for a Yee component at ``location``.

    Particle shapes are anchored on the cell-centered grid, so a component is
    staggered exactly on the axes its C/V location marks ``"V"``.  Dropping the
    gather order by one on those axes makes the gather the transpose of the
    Esirkepov scatter, which is what keeps the particle/field energy exchange
    consistent.

    Args:
        location (tuple): C/V triplet naming the component's Yee position.
        shape_order (int): The run's particle shape order.

    Returns:
        tuple: Shape order for the x, y and z axes.
    """
    return tuple(shape_order - 1 if axis == "V" else shape_order for axis in location)


def gather_shape_orders(location, static_parameters):
    """
    Gather orders for one component under the run's current deposition scheme.

    Galerkin gathering is the adjoint of Esirkepov deposition only.  The direct
    and ``GR_direct`` scatters are built against the full-order gather, so they
    keep the uniform shape order.
    """
    shape_order = static_parameters.shape_factor
    if static_parameters.current_deposition != "esirkepov":
        return shape_order
    return galerkin_ordering(location, shape_order)


def axis_shape_orders(shape_factor):
    """Accept one shape order for every axis, or one order per axis."""
    if isinstance(shape_factor, tuple):
        return shape_factor
    return (shape_factor, shape_factor, shape_factor)


@partial(jit, static_argnames=("ghost_cells", "active_axes", "inactive_axis_indices"))
def interpolate_field_to_particles(
    field,
    x,
    y,
    z,
    grid,
    shape_factor,
    ghost_cells=False,
    active_axes=None,
    inactive_axis_indices=None,
):
    """
    Interpolate a Yee-grid field to particle positions using PIC shape functions.

    Args:
        field (ndarray): Field values on the supplied component grid.  The
            first three axes are spatial; trailing component axes are
            interpolated together and preserved in the result.
        x, y, z (ndarray): Particle coordinates.
        grid (tuple): Component grid coordinates (x_grid, y_grid, z_grid).
        shape_factor (int or tuple): Particle shape order (0 -> NGP, 1 -> first
            order, 2 -> second order), either one order for every axis or a
            per-axis triplet as produced by :func:`galerkin_ordering`.

    Returns:
        ndarray: Interpolated field values with shape
            ``(n_particles,) + field.shape[3:]``.
    """
    x_grid, y_grid, z_grid = grid
    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)
    # grid point counts for each direction
    if active_axes is None:
        x_active = axis_has_active_cells(Nx, ghost_cells=ghost_cells)
        y_active = axis_has_active_cells(Ny, ghost_cells=ghost_cells)
        z_active = axis_has_active_cells(Nz, ghost_cells=ghost_cells)
    else:
        x_active, y_active, z_active = active_axes

    x_order, y_order, z_order = axis_shape_orders(shape_factor)
    # the Galerkin ordering gives a staggered axis its own reduced shape order

    _, _, deltax, xpts = prepare_particle_axis_stencil(
        x, x_grid, Nx, x_order, BC_PERIODIC, ghost_cells=ghost_cells
    )
    _, _, deltay, ypts = prepare_particle_axis_stencil(
        y, y_grid, Ny, y_order, BC_PERIODIC, ghost_cells=ghost_cells
    )
    _, _, deltaz, zpts = prepare_particle_axis_stencil(
        z, z_grid, Nz, z_order, BC_PERIODIC, ghost_cells=ghost_cells
    )
    dx = uniform_axis_spacing(x_grid)
    dy = uniform_axis_spacing(y_grid)
    dz = uniform_axis_spacing(z_grid)
    # determine the distance from the closest grid nodes

    x_weights = get_axis_weights(deltax, dx, x_order)
    y_weights = get_axis_weights(deltay, dy, y_order)
    z_weights = get_axis_weights(deltaz, dz, z_order)
    # compute the shape function weights for the particles on each axis

    xpts = jnp.asarray(xpts)
    ypts = jnp.asarray(ypts)
    zpts = jnp.asarray(zpts)

    def collapse_axis(points, weights, axis_size, axis_active, inactive_index):
        if axis_active:
            return points, weights

        if inactive_index is None:
            inactive_index = inactive_axis_index(axis_size, ghost_cells=ghost_cells)

        collapsed_points = jnp.full(
            (1, points.shape[1]),
            int(inactive_index),
            dtype=points.dtype,
        )
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights

    if inactive_axis_indices is None:
        inactive_axis_indices = (None, None, None)

    # Keep full shape-factor computation but collapse inactive axes to an
    # effective stencil size of 1 to avoid redundant interpolation work.
    if x_active:
        xpts_eff = xpts
        x_weights_eff = x_weights
    else:
        xpts_eff, x_weights_eff = collapse_axis(
            xpts, x_weights, Nx, x_active, inactive_axis_indices[0]
        )

    if y_active:
        ypts_eff = ypts
        y_weights_eff = y_weights
    else:
        ypts_eff, y_weights_eff = collapse_axis(
            ypts, y_weights, Ny, y_active, inactive_axis_indices[1]
        )

    if z_active:
        zpts_eff = zpts
        z_weights_eff = z_weights
    else:
        zpts_eff, z_weights_eff = collapse_axis(
            zpts, z_weights, Nz, z_active, inactive_axis_indices[2]
        )

    stencil_values = field[
        xpts_eff[:, jnp.newaxis, jnp.newaxis, :],
        ypts_eff[jnp.newaxis, :, jnp.newaxis, :],
        zpts_eff[jnp.newaxis, jnp.newaxis, :, :],
    ]
    interpolated_field = jnp.einsum(
        "ip,jp,kp,ijkp...->p...",
        x_weights_eff,
        y_weights_eff,
        z_weights_eff,
        stencil_values,
    )
    # Contract the three shape weights over the spatial stencil while keeping
    # particle and trailing field-component axes explicit.

    return interpolated_field
