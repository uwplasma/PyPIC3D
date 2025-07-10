import jax
import jax.numpy as jnp

from PyPIC3D.fdtd import (
    centered_finite_difference_curl,
    centered_finite_difference_laplacian,
    centered_finite_difference_divergence,
    centered_finite_difference_gradient
)
from PyPIC3D.boris import create_trilinear_interpolator, create_quadratic_interpolator


def initialize_vector_potential(J, world, constants):
    """
    Initialize the vector potential A based on the current density J.

    Args:
        J (tuple): Current density components (Jx, Jy, Jz).
        world (dict): Simulation world parameters including 'dx', 'dy', 'dz'.
        constants (dict): Physical constants including 'mu'.

    Returns:
        tuple: Initial vector potential components (Ax, Ay, Az).
    """
    Jx, Jy, Jz = J

    Ax = jnp.zeros_like(Jx)
    Ay = jnp.zeros_like(Jy)
    Az = jnp.zeros_like(Jz)
    # initialize A as zero

    A = Ax, Ay, Az

    return A, A, A

@jax.jit
def update_vector_potential(J, world, constants, A1, A0):
    """
    Updates the vector potential components (Ax, Ay, Az) using the finite difference method.

    This function advances the vector potential in time based on the current density,
    the previous two time steps of the vector potential, and physical constants. It uses
    centered finite difference schemes to compute the Laplacian, divergence, and gradient
    of the vector potential, assuming periodic boundary conditions.

    Parameters:
        J (tuple of ndarray): The current density components (Jx, Jy, Jz).
        world (dict): Simulation parameters, must contain 'dx', 'dy', 'dz', and 'dt'.
        constants (dict): Physical constants, must contain 'mu' (permeability) and 'C' (speed of light).
        A1 (tuple of ndarray): The vector potential components (Ax, Ay, Az) at the current time step.
        A0 (tuple of ndarray): The vector potential components (Ax0, Ay0, Az0) at the previous time step.

    Returns:
        tuple of ndarray: The updated vector potential components (Ax_new, Ay_new, Az_new).
    """

    Ax0, Ay0, Az0 = A0
    Ax, Ay, Az = A1
    Jx, Jy, Jz = J

    dx, dy, dz = world['dx'], world['dy'], world['dz']

    mu = constants['mu']
    C  = constants['C']
    dt = world['dt']

    laplacian_Ax = centered_finite_difference_laplacian(Ax, dx, dy, dz, 'periodic')
    laplacian_Ay = centered_finite_difference_laplacian(Ay, dx, dy, dz, 'periodic')
    laplacian_Az = centered_finite_difference_laplacian(Az, dx, dy, dz, 'periodic')
    # calculate the Laplacian of the vector potential using centered finite difference

    divA         = centered_finite_difference_divergence(Ax, Ay, Az, dx, dy, dz, 'periodic')
    # calculate the divergence of the vector potential using centered finite difference

    gradx, grady, gradz = centered_finite_difference_gradient(divA, dx, dy, dz, 'periodic')
    # calculate the gradient of the divergence of the vector potential using centered finite difference

    Ax_new = 2 * Ax - Ax0 + C**2 * dt**2 * ( mu * Jx  + laplacian_Ax - gradx )
    Ay_new = 2 * Ay - Ay0 + C**2 * dt**2 * ( mu * Jy  + laplacian_Ay - grady )
    Az_new = 2 * Az - Az0 + C**2 * dt**2 * ( mu * Jz  + laplacian_Az - gradz )
    # update the vector potential using centered finite difference

    return Ax_new, Ay_new, Az_new

@jax.jit
def E_from_A(A2, A0, world):
    """
    Calculate the electric field components from the vector potential using a centered finite difference.
    Args:
        A2 (tuple or list of ndarray): The vector potential components (Ax, Ay, Az) at the next time step.
        A0 (tuple or list of ndarray): The vector potential components (Ax0, Ay0, Az0) at the previous time step.
        world (dict): Dictionary containing simulation parameters. Must include the time step 'dt'.
    Returns:
        tuple of ndarray: The electric field components (Ex, Ey, Ez) computed from the vector potential.
    """

    Ax, Ay, Az = A2
    Ax0, Ay0, Az0 = A0
    dt = world['dt']

    Ex = -1 * (Ax - Ax0) / ( 2 * dt )
    Ey = -1 * (Ay - Ay0) / ( 2 * dt )
    Ez = -1 * (Az - Az0) / ( 2 * dt )
    # calculate the electric field from the vector potential using centered finite difference

    return Ex, Ey, Ez

@jax.jit
def B_from_A(A, world, E_grid, B_grid, interpolation_order=2):
    """
    Computes the magnetic field components (Bx, By, Bz) from the vector potential (A)
    using a centered finite difference curl and interpolates the result to the Yee grid.

    Parameters
    ----------
    A : tuple of ndarray
        The vector potential components (Ax, Ay, Az) as 3D arrays.
    world : dict
        Dictionary containing grid spacing with keys 'dx', 'dy', 'dz'.
    E_grid : tuple or ndarray
        The grid on which the electric field is defined.
    B_grid : tuple or ndarray
        The grid on which the magnetic field is to be interpolated.
    interpolation_order : int, optional
        The order of interpolation to use when mapping the magnetic field to the Yee grid (default is 2).

    Returns
    -------
    Bx, By, Bz : ndarray
        The interpolated magnetic field components on the Yee grid.
    """

    Ax, Ay, Az = A
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    Bx, By, Bz = centered_finite_difference_curl(Ax, Ay, Az, dx, dy, dz, 'periodic')

    Bx = interpolate_field(Bx, E_grid, B_grid, interpolation_order)
    By = interpolate_field(By, E_grid, B_grid, interpolation_order)
    Bz = interpolate_field(Bz, E_grid, B_grid, interpolation_order)
    # interpolate the magnetic field components to the yee grid

    return Bx, By, Bz

def interpolate_field(field, grid, target_grid, interpolation_order=1):
    """
    Interpolates a 3D field from a source grid onto a target grid using the specified interpolation order.

    Parameters
    ----------
    field : array-like
        The 3D array representing the field values on the source grid.
    grid : tuple of array-like
        The source grid coordinates as (x, y, z).
    target_grid : tuple of array-like
        The target grid coordinates as (x, y, z) onto which the field will be interpolated.
    interpolation_order : int, optional
        The order of interpolation to use: 1 for trilinear, 2 for quadratic. Default is 1 (trilinear).

    Returns
    -------
    interp_field : ndarray
        The interpolated field values on the target grid, with shape matching the meshgrid of target_grid.

    Notes
    -----
    - Assumes periodic boundary conditions for the interpolation.
    - Requires `jax` and `jnp` (JAX NumPy) for computation.
    - Helper functions `create_trilinear_interpolator` and `create_quadratic_interpolator` must be defined.
    """

    x, y, z = target_grid
    # Unpack target grid coordinates

    X_target, Y_target, Z_target = jnp.meshgrid(x, y, z, indexing='ij')
    # Create a meshgrid for target coordinates

    x_flat = X_target.flatten()
    y_flat = Y_target.flatten()
    z_flat = Z_target.flatten()
    # Flatten target coordinates for vectorized interpolation

    interp_flat = jax.lax.cond(
        interpolation_order == 1,
        lambda _: create_trilinear_interpolator(field, grid, periodic=True)(x_flat, y_flat, z_flat),
        lambda _: create_quadratic_interpolator(field, grid, periodic=True)(x_flat, y_flat, z_flat),
        operand=None
    )

    interp_field = interp_flat.reshape(X_target.shape)
    # Reshape back to the target grid shape

    return interp_field