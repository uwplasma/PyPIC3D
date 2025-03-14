import jax.numpy as jnp
# import external libraries

from PyPIC3D.utils import interpolate_field, use_gpu_if_set
from PyPIC3D.boundaryconditions import apply_zero_boundary_condition

import jax.numpy as jnp
#from lineax import Diagonal, Identity, Sum, Scale, Shift

#@partial(jit, static_argnums=(1, 2, 3, 4))
def centered_finite_difference_laplacian(field, dx, dy, dz, bc):
    """
    Calculates the Laplacian of a given field using centered finite difference and applies the specified boundary conditions.

    Args:
        field: numpy.ndarray
            The input field.
        dx: float
            The spacing between grid points in the x-direction.
        dy: float
            The spacing between grid points in the y-direction.
        dz: float
            The spacing between grid points in the z-direction.
        boundary_condition: str
            The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
        numpy.ndarray
            The Laplacian of the field with the specified boundary conditions applied.
    """

    if bc == 'dirichlet':
        field = apply_zero_boundary_condition(field)
        # apply zero boundary condition at the edges of the field

    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field) / (dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field) / (dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field) / (dz*dz)
    # calculate the Laplacian of the field using centered finite difference

    if bc == 'neumann':
        x_comp = apply_zero_boundary_condition(x_comp)
        y_comp = apply_zero_boundary_condition(y_comp)
        z_comp = apply_zero_boundary_condition(z_comp)

    return x_comp + y_comp + z_comp

#@partial(jit, static_argnums=(3, 4, 5, 6))
def centered_finite_difference_curl(field_x, field_y, field_z, dx, dy, dz, bc):
    """
    Computes the curl of a vector field using centered finite differencing and applies the specified boundary conditions.

    Args:
        field_x: numpy.ndarray
            The x-component of the vector field.
        field_y: numpy.ndarray
            The y-component of the vector field.
        field_z: numpy.ndarray
            The z-component of the vector field.
        dx: float
            The spacing between grid points in the x-direction.
        dy: float
            The spacing between grid points in the y-direction.
        dz: float
            The spacing between grid points in the z-direction.
        bc: str
            The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
        tuple of numpy.ndarray
            The curl components (curl_x, curl_y, curl_z) of the vector field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field_x = apply_zero_boundary_condition(field_x)
        field_y = apply_zero_boundary_condition(field_y)
        field_z = apply_zero_boundary_condition(field_z)

    dfx_dy = (jnp.roll(field_x, 1, axis=1) - jnp.roll(field_x, -1, axis=1)) / (2 * dy)
    dfx_dz = (jnp.roll(field_x, 1, axis=2) - jnp.roll(field_x, -1, axis=2)) / (2 * dz)
    # calculate the partial derivative of the x-component of the field with respect to y and z
    dfy_dx = (jnp.roll(field_y, 1, axis=2) - jnp.roll(field_y, -1, axis=2)) / (2 * dz)
    dfy_dz = (jnp.roll(field_y, 1, axis=0) - jnp.roll(field_y, -1, axis=0)) / (2 * dx)
    # calculate the partial derivative of the y-component of the field with respect to x and z
    dfz_dx = (jnp.roll(field_z, 1, axis=0) - jnp.roll(field_z, -1, axis=0)) / (2 * dx)
    dfz_dy = (jnp.roll(field_z, 1, axis=1) - jnp.roll(field_z, -1, axis=1)) / (2 * dy)
    # calculate the partial derivative of the z-component of the field with respect to x and y

    curl_x = dfz_dy - dfy_dz
    curl_y = dfx_dz - dfz_dx
    curl_z = dfy_dx - dfx_dy
    # calculate the curl of the field

    if bc == 'neumann':
        curl_x = apply_zero_boundary_condition(curl_x)
        curl_y = apply_zero_boundary_condition(curl_y)
        curl_z = apply_zero_boundary_condition(curl_z)

    return curl_x, curl_y, curl_z

#@partial(jit, static_argnums=(1, 2, 3, 4))
def centered_finite_difference_gradient(field, dx, dy, dz, bc):
    """
    Computes the gradient of a scalar field using centered finite differencing and applies the specified boundary conditions.

    Args:
        field: numpy.ndarray
            The input scalar field.
        dx: float
            The spacing between grid points in the x-direction.
        dy: float
            The spacing between grid points in the y-direction.
        dz: float
            The spacing between grid points in the z-direction.
        bc: str
            The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
        tuple of numpy.ndarray
            The gradient components (grad_x, grad_y, grad_z) of the scalar field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field = apply_zero_boundary_condition(field)

    grad_x = (jnp.roll(field, shift=-1, axis=0) - jnp.roll(field, shift=1, axis=0)) / (2 * dx)
    grad_y = (jnp.roll(field, shift=-1, axis=1) - jnp.roll(field, shift=1, axis=1)) / (2 * dy)
    grad_z = (jnp.roll(field, shift=-1, axis=2) - jnp.roll(field, shift=1, axis=2)) / (2 * dz)

    if bc == 'neumann':
        grad_x = apply_zero_boundary_condition(grad_x)
        grad_y = apply_zero_boundary_condition(grad_y)
        grad_z = apply_zero_boundary_condition(grad_z)

    return grad_x, grad_y, grad_z

#@partial(jit, static_argnums=(3, 4, 5, 6))
def centered_finite_difference_divergence(field_x, field_y, field_z, dx, dy, dz, bc):
    """
    Computes the divergence of a vector field using centered finite differencing and applies the specified boundary conditions.

    Args:
        field_x: numpy.ndarray
            The x-component of the vector field.
        field_y: numpy.ndarray
            The y-component of the vector field.
        field_z: numpy.ndarray
            The z-component of the vector field.
        dx: float
            The spacing between grid points in the x-direction.
        dy: float
            The spacing between grid points in the y-direction.
        dz: float
            The spacing between grid points in the z-direction.
        bc: str
            The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
        numpy.ndarray
            The divergence of the vector field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field_x = apply_zero_boundary_condition(field_x)
        field_y = apply_zero_boundary_condition(field_y)
        field_z = apply_zero_boundary_condition(field_z)

    div_x = (jnp.roll(field_x, shift=-1, axis=0) - jnp.roll(field_x, shift=1, axis=0)) / (2 * dx)
    div_y = (jnp.roll(field_y, shift=-1, axis=1) - jnp.roll(field_y, shift=1, axis=1)) / (2 * dy)
    div_z = (jnp.roll(field_z, shift=-1, axis=2) - jnp.roll(field_z, shift=1, axis=2)) / (2 * dz)

    if bc == 'neumann':
        div_x = apply_zero_boundary_condition(div_x)
        div_y = apply_zero_boundary_condition(div_y)
        div_z = apply_zero_boundary_condition(div_z)

    return div_x + div_y + div_z


# def finite_difference_laplacian_3d(Nx, Ny, Nz, dx, dy, dz, bc='dirichlet'):
#     """
#     Constructs the Laplacian operator in 3D dimensions using finite differencing.

#     Args:
#         Nx (int): Number of grid points in the x-direction.
#         Ny (int): Number of grid points in the y-direction.
#         Nz (int): Number of grid points in the z-direction.
#         dx (float): Grid spacing in the x-direction.
#         dy (float): Grid spacing in the y-direction.
#         dz (float): Grid spacing in the z-direction.
#         bc (str): Boundary condition type ('dirichlet' or 'neumann').

#     Returns:
#         LinearOperator: The Laplacian operator.
#     """
#     # Define the finite difference coefficients
#     coeff_x = -2.0 / (dx * dx)
#     coeff_y = -2.0 / (dy * dy)
#     coeff_z = -2.0 / (dz * dz)
#     coeff_xy = 1.0 / (dx * dx)
#     coeff_yz = 1.0 / (dy * dy)
#     coeff_zx = 1.0 / (dz * dz)

#     # Construct the Laplacian operator
#     laplacian = Sum([
#         Scale(coeff_x, Identity((Nx, Ny, Nz))),
#         Scale(coeff_y, Identity((Nx, Ny, Nz))),
#         Scale(coeff_z, Identity((Nx, Ny, Nz))),
#         Scale(coeff_xy, Shift((Nx, Ny, Nz), (1, 0, 0))),
#         Scale(coeff_xy, Shift((Nx, Ny, Nz), (-1, 0, 0))),
#         Scale(coeff_yz, Shift((Nx, Ny, Nz), (0, 1, 0))),
#         Scale(coeff_yz, Shift((Nx, Ny, Nz), (0, -1, 0))),
#         Scale(coeff_zx, Shift((Nx, Ny, Nz), (0, 0, 1))),
#         Scale(coeff_zx, Shift((Nx, Ny, Nz), (0, 0, -1))),
#     ])

#     # Apply boundary conditions
#     if bc == 'dirichlet':
#         # Apply Dirichlet boundary conditions (zero at the boundaries)
#         mask = jnp.ones((Nx, Ny, Nz))
#         mask = mask.at[0, :, :].set(0)
#         mask = mask.at[-1, :, :].set(0)
#         mask = mask.at[:, 0, :].set(0)
#         mask = mask.at[:, -1, :].set(0)
#         mask = mask.at[:, :, 0].set(0)
#         mask = mask.at[:, :, -1].set(0)
#         return Diagonal(mask) @ laplacian @ Diagonal(mask)
    
#     elif bc == 'neumann':
#         # Apply Neumann boundary conditions (zero gradient at the boundaries)
#         mask = jnp.ones((Nx, Ny, Nz))
#         mask = mask.at[0, :, :].set(0)
#         mask = mask.at[-1, :, :].set(0)
#         mask = mask.at[:, 0, :].set(0)
#         mask = mask.at[:, -1, :].set(0)
#         mask = mask.at[:, :, 0].set(0)
#         mask = mask.at[:, :, -1].set(0)
#         return Diagonal(mask) @ laplacian

#     return laplacian