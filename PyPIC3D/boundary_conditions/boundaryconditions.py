# Christopher Woolford, April 6th 2026
# Field-level boundary condition operations for ordinary 3D arrays.

from jax import jit
import jax.numpy as jnp
# import external libraries

# No @jit: the Python for-loop is unrolled during JAX tracing by any
# JIT-compiled caller, so explicit JIT here would cause tracing errors.
def apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength):
    """
    Apply a super-Gaussian absorbing boundary layer to damp fields near domain edges.

    Multiplies the field by an exponential damping profile that increases
    toward the boundary: exp(-strength * (distance / thickness)^order).
    This is applied symmetrically on all six faces of the 3D domain.

    Args:
        field (jnp.ndarray): 3D field array with shape (Nx, Ny, Nz).
        boundary_thickness (int): Number of cells in the absorbing layer.
        order (int): Exponent of the super-Gaussian profile (higher = sharper transition).
        strength (float): Amplitude of the damping (dimensionless).

    Returns:
        jnp.ndarray: Field with absorbing boundary damping applied.
    """
    def supergaussian_factor(x, thickness, order, strength):
        """Compute the damping factor at distance x from the boundary."""
        return jnp.exp(-strength * (x / thickness)**order)

    nx, ny, nz = field.shape
    for i in range(boundary_thickness):
        factor = supergaussian_factor(i, boundary_thickness, order, strength)
        field = field.at[i, :, :].mul(factor)
        field = field.at[nx - 1 - i, :, :].mul(factor)
        field = field.at[:, i, :].mul(factor)
        field = field.at[:, ny - 1 - i, :].mul(factor)
        field = field.at[:, :, i].mul(factor)
        field = field.at[:, :, nz - 1 - i].mul(factor)

    return field


@jit
def apply_zero_boundary_condition(field):
    """
    Zero all six boundary faces of a 3D field.

    Sets the outermost slice on each axis to zero. This is a simple
    Dirichlet-zero condition applied directly to the array boundaries,
    without distinguishing ghost cells from physical cells.

    Args:
        field (jnp.ndarray): 3D field array of any shape.

    Returns:
        jnp.ndarray: Field with all boundary faces set to zero.
    """
    field = field.at[0, :, :].set(0)
    field = field.at[-1, :, :].set(0)
    field = field.at[:, 0, :].set(0)
    field = field.at[:, -1, :].set(0)
    field = field.at[:, :, 0].set(0)
    field = field.at[:, :, -1].set(0)

    return field
