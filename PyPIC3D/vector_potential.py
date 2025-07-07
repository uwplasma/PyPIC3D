import jax
import jax.numpy as jnp

from PyPIC3D.fdtd import (
    centered_finite_difference_curl,
    centered_finite_difference_laplacian,
    centered_finite_difference_divergence,
    centered_finite_difference_gradient
)


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
    Update the vector potential A based on the magnetic field B using centered finite difference.

    Args:
        A (tuple): Vector potential components (Ax, Ay, Az).
        J (tuple): Current density components (Jx, Jy, Jz).
        dt (float): Time step.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        bc (str): Boundary condition type ('periodic', 'neumann', 'dirichlet').

    Returns:
        tuple: Updated vector potential components (Ax_new, Ay_new, Az_new).
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

    Ax_new = 2 * Ax - Ax0 + C**2 * dt**2 * ( mu * Jx  + laplacian_Ax + gradx )
    Ay_new = 2 * Ay - Ay0 + C**2 * dt**2 * ( mu * Jy  + laplacian_Ay + grady )
    Az_new = 2 * Az - Az0 + C**2 * dt**2 * ( mu * Jz  + laplacian_Az + gradz )
    # update the vector potential using centered finite difference

    return Ax_new, Ay_new, Az_new

@jax.jit
def E_from_A(A2, A0, world):
    Ax, Ay, Az = A2
    Ax0, Ay0, Az0 = A0
    dt = world['dt']

    Ex = -1 * (Ax - Ax0) / (2 * dt )
    Ey = -1 * (Ay - Ay0) / (2 * dt )
    Ez = -1 * (Az - Az0) / (2 * dt )
    # calculate the electric field from the vector potential using centered finite difference

    return Ex, Ey, Ez

@jax.jit
def B_from_A(A, world):
    """
    Calculate the magnetic field B from the vector potential A using centered finite difference.

    Args:
        A (tuple): Vector potential components (Ax, Ay, Az).
        world (dict): Simulation world parameters including 'dx', 'dy', 'dz'.

    Returns:
        tuple: Magnetic field components (Bx, By, Bz).
    """
    Ax, Ay, Az = A
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    Bx, By, Bz = centered_finite_difference_curl(Ax, Ay, Az, dx, dy, dz, 'periodic')

    return Bx, By, Bz