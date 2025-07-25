import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
import functools
from functools import partial
# import external libraries

from PyPIC3D.solvers.pstd import spectral_poisson_solve, spectral_gradient
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient, centered_finite_difference_curl
from PyPIC3D.rho import compute_rho
from PyPIC3D.utils import digital_filter
# import internal libraries

@partial(jit, static_argnames=("curl_func"))
def update_E(E, B, J, world, constants, curl_func):
    """
    Update the electric field components (Ex, Ey, Ez) based on the given parameters.

    Args:
        grid (object): The grid object containing the simulation grid.
        staggered_grid (object): The staggered grid object for the simulation.
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
    C = constants['C']
    eps = constants['eps']
    # get the time resolution and necessary constants

    curlx, curly, curlz = curl_func(Bx, By, Bz)
    # calculate the curl of the magnetic field

    Ex = Ex + ( 2 * C**2 * curlx - Jx / eps ) * dt
    Ey = Ey + ( 2 * C**2 * curly - Jy / eps ) * dt
    Ez = Ez + ( 2 * C**2 * curlz - Jz / eps ) * dt
    # update the electric field from Maxwell's equations

    alpha = constants['alpha']
    Ex = digital_filter(Ex, alpha)
    Ey = digital_filter(Ey, alpha)
    Ez = digital_filter(Ez, alpha)
    # apply a digital filter to the electric field components

    return (Ex, Ey, Ez)


@partial(jit, static_argnames=("curl_func"))
def update_B(E, B, world, constants, curl_func):
    """
    Update the magnetic field components (Bx, By, Bz) using the curl of the electric field.

    Args:
        grid (ndarray): The grid on which the fields are defined.
        staggered_grid (ndarray): The staggered grid for field calculations.
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

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # unpack the E and B fields

    curlx, curly, curlz = curl_func(Ex, Ey, Ez)
    # calculate the curl of the electric field

    Bx = Bx - 2*dt*curlx
    By = By - 2*dt*curly
    Bz = Bz - 2*dt*curlz
    # update the magnetic field from Maxwell's equations

    alpha = constants['alpha']
    Bx = digital_filter(Bx, alpha)
    By = digital_filter(By, alpha)
    Bz = digital_filter(Bz, alpha)
    # apply a digital filter to the magnetic field components

    return (Bx, By, Bz)


@partial(jit, static_argnames=("solver", "bc"))
def calculateE(world, particles, constants, rho, phi, solver, bc):
    """
    Calculate the electric field components (Ex, Ey, Ez) and electric potential (phi)
    based on the given parameters.

    Args:
        E (tuple): Tuple containing the electric field components (Ex, Ey, Ez).
        world (dict): Dictionary containing the simulation world parameters such as
                    grid spacing (dx, dy, dz) and window dimensions (x_wind, y_wind, z_wind).
        particles (array): Array containing particle positions and properties.
        constants (dict): Dictionary containing physical constants such as permittivity (eps).
        rho (array): Charge density array.
        phi (array): Electric potential array.
        solver (str): Type of solver to use ('spectral' or other).
        bc (str): Boundary condition type.
        verbose (bool): Flag to enable verbose output.

    Returns:
        tuple: Updated electric field components (Ex, Ey, Ez), electric potential (phi),
            and charge density (rho).
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get resolution

    rho = compute_rho(particles, rho, world, constants)
    # calculate the charge density based on the particle positions

    phi = spectral_poisson_solve(rho, constants, world)
    # solve the Poisson equation to get the electric potential

    alpha = constants['alpha']
    phi = digital_filter(phi, alpha)
    # apply a digital filter to the electric potential
    # alpha = 0.5 is a bilinear filter for the electric potential

    Ex, Ey, Ez = lax.cond(
        solver == 'spectral',
        lambda _: spectral_gradient(-1*phi, world),
        lambda _: centered_finite_difference_gradient(-1*phi, dx, dy, dz, bc),
        operand=None
    )
    # compute the gradient of the electric potential to get the electric field

    return (Ex, Ey, Ez), phi, rho