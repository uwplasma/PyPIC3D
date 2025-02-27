from jax import jit
import jax.numpy as jnp
from functools import partial
# import external libraries

from PyPIC3D.pstd import spectral_laplacian, spectral_divergence
from PyPIC3D.fdtd import centered_finite_difference_laplacian, centered_finite_difference_divergence
# import functions from the PyPIC3D package

@partial(jit, static_argnums=(4, 5))
def compute_pe(phi, rho, constants, world, solver, bc='periodic'):
    """
    Compute the relative percentage difference of the Poisson solver.

    Args:
        phi (ndarray): The potential field.
        rho (ndarray): The charge density.
        eps (float): The permittivity.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.
        bc (str): The boundary condition.

    Returns:
        float: The relative percentage difference of the Poisson solver.
    """
    eps = constants['eps']
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    if solver == 'spectral':
        x = spectral_laplacian(phi, world)
    elif solver == 'fdtd':
        x = centered_finite_difference_laplacian(phi, dx, dy, dz, bc)
    elif solver == 'autodiff':
        return 0
    poisson_error = x + rho/eps
    magnitude = jnp.mean(jnp.abs(rho/eps)) + 1e-16
    return jnp.mean(jnp.abs(poisson_error)) / magnitude

@partial(jit, static_argnums=(4, 5))
def compute_magnetic_divergence_error(Bx, By, Bz, world, solver, bc='periodic'):
    """
    Compute the error in the divergence of the magnetic field for different solvers.

    Args:
        Bx (ndarray): The x-component of the magnetic field.
        By (ndarray): The y-component of the magnetic field.
        Bz (ndarray): The z-component of the magnetic field.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.
        bc (str): The boundary condition.

    Returns:
        float: The error in the divergence of the magnetic field.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    if solver == 'spectral':
        divB = spectral_divergence(Bx, By, Bz, world)
    elif solver == 'fdtd':
        divB = centered_finite_difference_divergence(Bx, By, Bz, dx, dy, dz, bc)
    elif solver == 'autodiff':
        return 0
    divergence_error = jnp.mean(jnp.abs(divB))
    return divergence_error

@partial(jit, static_argnums=(6, 7))
def compute_electric_divergence_error(Ex, Ey, Ez, rho, constants, world, solver, bc='periodic'):
    """
    Compute the error in the divergence of the electric field using the charge density and the components of the electric field.

    Args:
        Ex (ndarray): The x-component of the electric field.
        Ey (ndarray): The y-component of the electric field.
        Ez (ndarray): The z-component of the electric field.
        rho (ndarray): The charge density.
        eps (float): The permittivity.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.
        bc (str): The boundary condition.

    Returns:
    float: The error in the divergence of the electric field.
    """
    eps = constants['eps']
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    if solver == 'spectral':
        divE = spectral_divergence(Ex, Ey, Ez, world)
    elif solver == 'fdtd':
        divE = centered_finite_difference_divergence(Ex, Ey, Ez, dx, dy, dz, bc)
    elif solver == 'autodiff':
        return 0

    divergence_error = jnp.mean(jnp.abs(divE - rho / eps))
    return divergence_error