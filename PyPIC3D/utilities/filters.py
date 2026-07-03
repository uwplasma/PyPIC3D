import jax.numpy as jnp
from functools import partial
from jax import jit, lax, vmap


def apply_valid_3d_conv_last_axes(phi, kernel_3d):
    """
    Apply a 3D VALID convolution over the last three axes of phi.

    phi shape:
        (..., Nx + 2, Ny + 2, Nz + 2)

    kernel_3d shape:
        (3, 3, 3)

    Returns:
        filtered interior with shape (..., Nx, Ny, Nz)
    """
    leading_shape = phi.shape[:-3]
    spatial_shape = phi.shape[-3:]

    # Flatten all leading dimensions into a batch axis.
    x = phi.reshape((-1, 1, *spatial_shape))

    # lax.conv_general_dilated expects:
    # lhs:    (N, C, D, H, W)
    # rhs:    (O, I, D, H, W)
    # output: (N, C, D, H, W)
    kernel = kernel_3d[None, None, :, :, :]

    filtered = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1, 1),
        padding="VALID",
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
    )

    # Remove channel dimension and restore leading dimensions.
    filtered = filtered[:, 0]

    return filtered.reshape(
        (*leading_shape, spatial_shape[0] - 2, spatial_shape[1] - 2, spatial_shape[2] - 2)
    )

def _active_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g, -g)


def _stencil_slice(num_guard_cells):
    g = int(num_guard_cells)
    return slice(g - 1, None if g == 1 else -g + 1)


def _is_stacked_vector_field(field):
    return hasattr(field, "ndim") and field.ndim >= 4 and int(field.shape[0]) == 3


def _stack_vector_field(field):
    if _is_stacked_vector_field(field):
        return field
    return jnp.stack(field, axis=0)


def _restore_vector_field(stacked_field, original_field):
    if _is_stacked_vector_field(original_field):
        return stacked_field
    return stacked_field[0], stacked_field[1], stacked_field[2]


@partial(jit, static_argnames=("num_guard_cells",))
def bilinear_filter(phi, num_guard_cells=1):
    """
    Apply a 3D tri-linear smoothing filter to a ghost-celled field.

    The last three axes are spatial.  ``num_guard_cells`` selects the physical
    interior ``g:-g``; the filter reads the one-cell stencil halo around that
    interior and leaves all guard cells unchanged.
    """
    k1 = jnp.array([1.0, 2.0, 1.0], dtype=phi.dtype)

    kernel_3d = (
        k1[:, None, None]
        * k1[None, :, None]
        * k1[None, None, :]
    ) / 64.0

    stencil = _stencil_slice(num_guard_cells)
    filtered = apply_valid_3d_conv_last_axes(phi[..., stencil, stencil, stencil], kernel_3d)

    active = _active_slice(num_guard_cells)

    return phi.at[..., active, active, active].set(filtered)


@partial(jit, static_argnames=("num_guard_cells",))
def digital_filter(phi, alpha, num_guard_cells=1):
    """
    Apply a 3D nearest-neighbor digital filter to a ghost-celled field.

    The last three axes are spatial.  ``num_guard_cells`` selects the physical
    interior ``g:-g``; the filter reads the one-cell stencil halo around that
    interior and leaves all guard cells unchanged.
    """
    neighbor_weight = (1.0 - alpha) / 6.0

    kernel_3d = jnp.zeros((3, 3, 3), dtype=phi.dtype)

    # Center
    kernel_3d = kernel_3d.at[1, 1, 1].set(alpha)

    # Face neighbors
    kernel_3d = kernel_3d.at[0, 1, 1].set(neighbor_weight)
    kernel_3d = kernel_3d.at[2, 1, 1].set(neighbor_weight)

    kernel_3d = kernel_3d.at[1, 0, 1].set(neighbor_weight)
    kernel_3d = kernel_3d.at[1, 2, 1].set(neighbor_weight)

    kernel_3d = kernel_3d.at[1, 1, 0].set(neighbor_weight)
    kernel_3d = kernel_3d.at[1, 1, 2].set(neighbor_weight)

    stencil = _stencil_slice(num_guard_cells)
    filtered = apply_valid_3d_conv_last_axes(phi[..., stencil, stencil, stencil], kernel_3d)
    # Apply the digital filter to the interior of phi using a 3D convolution with the specified kernel.

    active = _active_slice(num_guard_cells)

    return phi.at[..., active, active, active].set(filtered)


def bilinear_filter_vector(field, num_guard_cells=1):
    """
    Apply the tri-linear filter component-wise to a vector field.
    """
    stacked = _stack_vector_field(field)
    filtered = vmap(
        lambda component: bilinear_filter(component, num_guard_cells=num_guard_cells),
        in_axes=0,
        out_axes=0,
    )(stacked)
    return _restore_vector_field(filtered, field)


def digital_filter_vector(field, alpha, num_guard_cells=1):
    """
    Apply the six-neighbor digital filter component-wise to a vector field.
    """
    stacked = _stack_vector_field(field)
    filtered = vmap(
        lambda component: digital_filter(component, alpha, num_guard_cells=num_guard_cells),
        in_axes=0,
        out_axes=0,
    )(stacked)
    return _restore_vector_field(filtered, field)
