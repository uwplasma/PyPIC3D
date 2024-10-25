from utils import autodiff_curlx, autodiff_curly, autodiff_curlz

# Christopher Woolford, October 24, 2024
# This script contains a seperate formulation of the particle in cell method using automatic differentiation

@jit
def autodiff_update_B(Bx, By, Bz, Ex, Ey, Ez, dt):
    """
    Update the magnetic field components (Bx, By, Bz) using the electric field components (Ex, Ey, Ez) and time step (dt).

    Parameters:
    Bx (ndarray): The x-component of the magnetic field.
    By (ndarray): The y-component of the magnetic field.
    Bz (ndarray): The z-component of the magnetic field.
    Ex (ndarray): The x-component of the electric field.
    Ey (ndarray): The y-component of the electric field.
    Ez (ndarray): The z-component of the electric field.
    dt (float): The time step for the update.

    Returns:
    tuple: Updated magnetic field components (Bx, By, Bz).
    """

    curlx, curly, curlz = autodiff_curlx(Ex, Ey, Ez)
    Bx = Bx - dt/2*curlx
    By = By - dt/2*curly
    Bz = Bz - dt/2*curlz
    # update the magnetic field
    return Bx, By, Bz

@jit
def autodiff_update_E(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dt, eps):
    """
    Update the electric field components Ex, Ey, and Ez based on the magnetic field components Bx, By, and Bz.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - Jx (ndarray): The x-component of the current density.
    - Jy (ndarray): The y-component of the current density.
    - Jz (ndarray): The z-component of the current density.
    - dt (float): The time step.
    - eps (float): The permittivity of the medium.

    Returns:
    - Ex (ndarray): The updated x-component of the electric field.
    - Ey (ndarray): The updated y-component of the electric field.
    - Ez (ndarray): The updated z-component of the electric field.
    """
    curlx, curly, curlz = autodiff_curlx(Bx, By, Bz)
    Ex = Ex + (curlx - Jx / eps) * dt / 2
    Ey = Ey + (curly - Jy / eps) * dt / 2
    Ez = Ez + (curlz - Jz / eps) * dt / 2
    return Ex, Ey, Ez