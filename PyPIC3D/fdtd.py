import jax.numpy as jnp
import jax
from jax import jit
from jax import lax
# import external libraries

from PyPIC3D.boundaryconditions import apply_zero_boundary_condition

def solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=1.5, tol=1e-6, max_iter=10000):
    """
    Solve Poisson's equation using Successive Over-Relaxation (SOR) method.

    Args:
        phi (jax.numpy.ndarray): Initial guess for the potential.
        rho (jax.numpy.ndarray): Charge density.
        omega (float): Relaxation factor.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        jax.numpy.ndarray: Solution for the potential.
    """
    phi = jnp.array(phi)
    rho = jnp.array(rho)
    b = dx * dy * dz / jnp.pi

    def cond_fun(val):
        _, norm_diff, iter_count = val
        return (norm_diff >= tol) & (iter_count < max_iter)

    def body_fun(val):
        phi, _, iter_count = val
        phi_old = phi.copy()
        phi = phi.at[1:-1, 1:-1, 1:-1].set(
            (1 - omega) * phi[1:-1, 1:-1, 1:-1] + omega / 6 * (
                phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
                phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
                phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:] -
                b * rho[1:-1, 1:-1, 1:-1] / eps
            )
        )
        return phi, jnp.linalg.norm(phi - phi_old), iter_count + 1

    phi, _, _ = jax.lax.while_loop(cond_fun, body_fun, (phi, jnp.inf, 0))

    return phi


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

    dFz_dy = (jnp.roll(field_z, shift=-1, axis=1) - jnp.roll(field_z, shift=1, axis=1)) / (2 * dy)
    dFy_dz = (jnp.roll(field_y, shift=-1, axis=2) - jnp.roll(field_y, shift=1, axis=2)) / (2 * dz)
    dFx_dz = (jnp.roll(field_x, shift=-1, axis=2) - jnp.roll(field_x, shift=1, axis=2)) / (2 * dz)
    dFz_dx = (jnp.roll(field_z, shift=-1, axis=0) - jnp.roll(field_z, shift=1, axis=0)) / (2 * dx)
    dFy_dx = (jnp.roll(field_y, shift=-1, axis=0) - jnp.roll(field_y, shift=1, axis=0)) / (2 * dx)
    dFx_dy = (jnp.roll(field_x, shift=-1, axis=1) - jnp.roll(field_x, shift=1, axis=1)) / (2 * dy)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
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