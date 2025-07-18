import jax
import jax.numpy as jnp

from PyPIC3D.solvers.fdtd import (
    centered_finite_difference_curl,
    centered_finite_difference_laplacian,
    centered_finite_difference_divergence,
    centered_finite_difference_gradient
)

@jax.jit
def update_B_second_order(B1, B0, J, E_grid, B_grid, world, constants):
    """
    Update the magnetic field B using the second-order centered finite difference method.

    Args:
        B1 (tuple): Current magnetic field components (Bx, By, Bz).
        B0 (tuple): Previous magnetic field components (Bx0, By0, Bz0).
        J (tuple): Current density components (Jx, Jy, Jz).
        world (dict): Simulation world parameters including 'dx', 'dy', 'dz'.
        constants (dict): Physical constants including 'mu'.

    Returns:
        tuple: Updated magnetic field components (Bx_new, By_new, Bz_new).
    """
    Bx1, By1, Bz1 = B1
    Bx0, By0, Bz0 = B0
    Jx, Jy, Jz = J

    dx, dy, dz = world['dx'], world['dy'], world['dz']
    dt  = world['dt']
    mu = constants['mu']
    C = constants['C']
    # extract the grid spacings and constants


    laplacian_Bx = centered_finite_difference_laplacian(Bx1, dx, dy, dz, 'periodic')
    laplacian_By = centered_finite_difference_laplacian(By1, dx, dy, dz, 'periodic')
    laplacian_Bz = centered_finite_difference_laplacian(Bz1, dx, dy, dz, 'periodic')
    # calculate the Laplacian of the magnetic field components

    curl_Jx, curl_Jy, curl_Jz = centered_finite_difference_curl(Jx, Jy, Jz, dx, dy, dz, 'periodic')
    # calculate the curl of the current density components

    Bx2 = 2*Bx1 - Bx0 + C**2 * dt**2 * (laplacian_Bx + mu*curl_Jx)
    By2 = 2*By1 - By0 + C**2 * dt**2 * (laplacian_By + mu*curl_Jy)
    Bz2 = 2*Bz1 - Bz0 + C**2 * dt**2 * (laplacian_Bz + mu*curl_Jz)
    # update the magnetic field components

    return Bx2, By2, Bz2


@jax.jit
def update_E_second_order(E1, E0, J1, J0, world, constants):
    """
    Update the electric field E using the second-order centered finite difference method.

    Args:
        E1 (tuple): Current electric field components (Ex, Ey, Ez).
        E0 (tuple): Previous electric field components (Ex0, Ey0, Ez0).
        J1 (tuple): Current density components (Jx, Jy, Jz).
        J0 (tuple): Previous density components (Jx0, Jy0, Jz0).
        world (dict): Simulation world parameters including 'dx', 'dy', 'dz'.
        constants (dict): Physical constants including 'epsilon'.

    Returns:
        tuple: Updated electric field components (Ex_new, Ey_new, Ez_new).
    """
    Ex1, Ey1, Ez1 = E1
    Ex0, Ey0, Ez0 = E0
    Jx1, Jy1, Jz1 = J1
    Jx0, Jy0, Jz0 = J0

    dx, dy, dz = world['dx'], world['dy'], world['dz']
    dt  = world['dt']
    C = constants['C']
    mu = constants['mu']
    # extract the grid spacings and constants

    laplacian_Ex = centered_finite_difference_laplacian(Ex1, dx, dy, dz, 'periodic')
    laplacian_Ey = centered_finite_difference_laplacian(Ey1, dx, dy, dz, 'periodic')
    laplacian_Ez = centered_finite_difference_laplacian(Ez1, dx, dy, dz, 'periodic')
    # calculate the Laplacian of the electric field components

    dJx_dt = (Jx1 - Jx0) / dt
    dJy_dt = (Jy1 - Jy0) / dt
    dJz_dt = (Jz1 - Jz0) / dt
    # calculate the time derivative of the current density components

    divE         = centered_finite_difference_divergence(Ex1, Ey1, Ez1, dx, dy, dz, 'periodic')
    # calculate the divergence of the vector potential using centered finite difference

    gradx, grady, gradz = centered_finite_difference_gradient(divE, dx, dy, dz, 'periodic')
    # calculate the gradient of the divergence of the vector potential using centered finite difference


    Ex2 = 2*Ex1 - Ex0 + C**2 * dt**2 * (laplacian_Ex - mu*dJx_dt - gradx)
    Ey2 = 2*Ey1 - Ey0 + C**2 * dt**2 * (laplacian_Ey - mu*dJy_dt - grady)
    Ez2 = 2*Ez1 - Ez0 + C**2 * dt**2 * (laplacian_Ez - mu*dJz_dt - gradz)
    # update the electric field components

    return Ex2, Ey2, Ez2