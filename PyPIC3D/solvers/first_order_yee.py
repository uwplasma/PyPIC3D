import jax.numpy as jnp
from jax import jit
from functools import partial
# import external libraries

from PyPIC3D.utils import digital_filter
from PyPIC3D.boundaryconditions import update_ghost_cells, apply_conducting_bc
# import internal libraries


@partial(jit, static_argnames=("curl_func",))
def update_E(E, B, J, world, constants, curl_func):
    """
    Update the electric field components (Ex, Ey, Ez) based on the given parameters.

    Uses ghost cells for boundary handling. Fields have shape (Nx+2, Ny+2, Nz+2)
    with 1 ghost cell on each side. The physical interior is [1:-1, 1:-1, 1:-1].

    Args:
        E (tuple): A tuple containing the electric field components (Ex, Ey, Ez).
        B (tuple): A tuple containing the magnetic field components (Bx, By, Bz).
        J (tuple): A tuple containing the current density components (Jx, Jy, Jz).
        world (dict): A dictionary containing the world parameters such as 'dx', 'dy', 'dz', and 'dt'.
        constants (dict): A dictionary containing the physical constants such as 'C' (speed of light) and 'eps' (permittivity).
        curl_func (function): A function to calculate the curl of the magnetic field.

    Returns:
        tuple: Updated electric field components (Ex, Ey, Ez).
    """

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J
    # unpack the E, B, and J fields

    dt = world['dt']
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
    # get the time resolution and grid spacings

    C = constants['C']
    eps = constants['eps']
    # get the necessary constants

    # forward differences using ghost cells
    # B has shape (Nx+2, Ny+2, Nz+2) with valid ghost cell values
    # slicing [1:-1, 2:, 1:-1] reaches into the ghost cell at the +1 boundary
    dBz_dy = (Bz[1:-1, 2:, 1:-1] - Bz[1:-1, 1:-1, 1:-1]) / dy
    dBy_dz = (By[1:-1, 1:-1, 2:] - By[1:-1, 1:-1, 1:-1]) / dz
    dBx_dz = (Bx[1:-1, 1:-1, 2:] - Bx[1:-1, 1:-1, 1:-1]) / dz
    dBx_dy = (Bx[1:-1, 2:, 1:-1] - Bx[1:-1, 1:-1, 1:-1]) / dy
    dBz_dx = (Bz[2:, 1:-1, 1:-1] - Bz[1:-1, 1:-1, 1:-1]) / dx
    dBy_dx = (By[2:, 1:-1, 1:-1] - By[1:-1, 1:-1, 1:-1]) / dx
    # compute the forward finite differences of the magnetic field

    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy
    # calculate the curl of the magnetic field

    Ex = Ex.at[1:-1, 1:-1, 1:-1].set(
        Ex[1:-1, 1:-1, 1:-1] + (C**2 * curl_x - Jx[1:-1, 1:-1, 1:-1] / eps) * dt
    )
    Ey = Ey.at[1:-1, 1:-1, 1:-1].set(
        Ey[1:-1, 1:-1, 1:-1] + (C**2 * curl_y - Jy[1:-1, 1:-1, 1:-1] / eps) * dt
    )
    Ez = Ez.at[1:-1, 1:-1, 1:-1].set(
        Ez[1:-1, 1:-1, 1:-1] + (C**2 * curl_z - Jz[1:-1, 1:-1, 1:-1] / eps) * dt
    )
    # update the electric field from Maxwell's equations

    alpha = constants['alpha']
    Ex = digital_filter(Ex, alpha)
    Ey = digital_filter(Ey, alpha)
    Ez = digital_filter(Ez, alpha)
    # apply a digital filter to the electric field components

    Ex, Ey, Ez = apply_conducting_bc((Ex, Ey, Ez), bc_x, bc_y, bc_z)
    # enforce conducting boundary conditions on tangential E

    Ex = update_ghost_cells(Ex, bc_x, bc_y, bc_z)
    Ey = update_ghost_cells(Ey, bc_x, bc_y, bc_z)
    Ez = update_ghost_cells(Ez, bc_x, bc_y, bc_z)
    # fill ghost cells with the updated values

    return (Ex, Ey, Ez)


@partial(jit, static_argnames=("curl_func",))
def update_B(E, B, world, constants, curl_func):
    """
    Update the magnetic field components (Bx, By, Bz) using the curl of the electric field.

    Uses ghost cells for boundary handling. Fields have shape (Nx+2, Ny+2, Nz+2)
    with 1 ghost cell on each side. The physical interior is [1:-1, 1:-1, 1:-1].

    Args:
        E (tuple): The electric field components (Ex, Ey, Ez).
        B (tuple): The magnetic field components (Bx, By, Bz).
        world (dict): Dictionary containing simulation parameters such as 'dx', 'dy', 'dz', and 'dt'.
        constants (dict): Dictionary containing physical constants.
        curl_func (function): Function to calculate the curl of the electric field.

    Returns:
        tuple: Updated magnetic field components (Bx, By, Bz).
    """

    dt = world['dt']
    # get the time resolution
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the grid spacings
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # unpack the E and B fields

    # backward differences using ghost cells
    # slicing [1:-1, :-2, 1:-1] reaches into the ghost cell at the -1 boundary
    dEz_dy = (Ez[1:-1, 1:-1, 1:-1] - Ez[1:-1, :-2, 1:-1]) / dy
    dEy_dz = (Ey[1:-1, 1:-1, 1:-1] - Ey[1:-1, 1:-1, :-2]) / dz
    dEx_dz = (Ex[1:-1, 1:-1, 1:-1] - Ex[1:-1, 1:-1, :-2]) / dz
    dEx_dy = (Ex[1:-1, 1:-1, 1:-1] - Ex[1:-1, :-2, 1:-1]) / dy
    dEz_dx = (Ez[1:-1, 1:-1, 1:-1] - Ez[:-2, 1:-1, 1:-1]) / dx
    dEy_dx = (Ey[1:-1, 1:-1, 1:-1] - Ey[:-2, 1:-1, 1:-1]) / dx
    # compute the backward finite differences of the electric field

    curl_x = dEz_dy - dEy_dz
    curl_y = dEx_dz - dEz_dx
    curl_z = dEy_dx - dEx_dy
    # calculate the curl of the electric field

    Bx = Bx.at[1:-1, 1:-1, 1:-1].set(Bx[1:-1, 1:-1, 1:-1] - dt * curl_x)
    By = By.at[1:-1, 1:-1, 1:-1].set(By[1:-1, 1:-1, 1:-1] - dt * curl_y)
    Bz = Bz.at[1:-1, 1:-1, 1:-1].set(Bz[1:-1, 1:-1, 1:-1] - dt * curl_z)
    # update the magnetic field from Maxwell's equations

    alpha = constants['alpha']
    Bx = digital_filter(Bx, alpha)
    By = digital_filter(By, alpha)
    Bz = digital_filter(Bz, alpha)
    # apply a digital filter to the magnetic field components

    Bx = update_ghost_cells(Bx, bc_x, bc_y, bc_z)
    By = update_ghost_cells(By, bc_x, bc_y, bc_z)
    Bz = update_ghost_cells(Bz, bc_x, bc_y, bc_z)
    # fill ghost cells with the updated values

    return (Bx, By, Bz)
