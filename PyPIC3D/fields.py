import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
import functools
from functools import partial
#from memory_profiler import profile
# import external libraries

from PyPIC3D.pstd import spectral_poisson_solve, spectral_gradient
from PyPIC3D.fdtd import centered_finite_difference_gradient, solve_poisson_sor
from PyPIC3D.rho import compute_rho
#from PyPIC3D.sor import solve_poisson_sor
# import internal libraries

def initialize_fields(Nx, Ny, Nz):
    """
    Initializes the electric and magnetic field arrays, as well as the electric potential and charge density arrays.

    Args:
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        Nz (int): Number of grid points in the z-direction.

    Returns:
        Ex (ndarray): Electric field array in the x-direction.
        Ey (ndarray): Electric field array in the y-direction.
        Ez (ndarray): Electric field array in the z-direction.
        Bx (ndarray): Magnetic field array in the x-direction.
        By (ndarray): Magnetic field array in the y-direction.
        Bz (ndarray): Magnetic field array in the z-direction.
        phi (ndarray): Electric potential array.
        rho (ndarray): Charge density array.
    """
    # get the number of grid points in each direction
    Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric field arrays as 0
    Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the magnetic field arrays as 0

    Jx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jy = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the current density arrays as 0

    phi = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric potential and charge density arrays as 0

    return (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), phi, rho

@partial(jit, static_argnums=(4, 5))
def solve_poisson(rho, constants, world, phi, solver, bc='periodic'):
    """
    Solve the Poisson equation for electrostatic potential.

    Args:
        rho (ndarray): Charge density.
        eps (float): Permittivity.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        phi (ndarray): Initial guess for the electrostatic potential.
        bc (str): Boundary condition.
        M (ndarray, optional): Preconditioner matrix for the conjugate gradient solver.

    Returns:
        phi (ndarray): Solution to the Poisson equation.
    """

    # phi = lax.cond(

    #     solver == 'spectral',

    #     lambda _: spectral_poisson_solve(rho, constants, world),
    #     lambda _: solve_poisson_sor(
    #         phi=phi,
    #         rho=rho,
    #         dx=world['dx'],
    #         dy=world['dy'],
    #         dz=world['dz'],
    #         eps=constants['eps'],
    #         omega=0.15,
    #         tol=1e-12,
    #         max_iter=15000
    #     ),

    #     operand=None
    # )

    phi = spectral_poisson_solve(rho, constants, world)

    # if solver == 'spectral':
    #     phi = spectral_poisson_solve()
    # elif solver == 'fdtd':
    #     phi = solve_poisson_sor()

    return phi

@partial(jit, static_argnums=(5, 6))
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

    rho = compute_rho(particles, rho, world)
    # calculate the charge density based on the particle positions

    phi = solve_poisson(rho=rho, constants=constants, world=world, phi=phi, solver=solver, bc=bc)
    # solve the Poisson equation to get the electric potential

    phi = digital_filter(phi, 0.5)
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


@partial(jit, static_argnums=(5))
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

    Ex = Ex + ( C**2 * curlx - Jx / eps ) * dt
    Ey = Ey + ( C**2 * curly - Jy / eps ) * dt
    Ez = Ez + ( C**2 * curlz - Jz / eps ) * dt
    # update the electric field from Maxwell's equations

    return (Ex, Ey, Ez)


@partial(jit, static_argnums=(4))
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

    Bx = Bx - dt*curlx
    By = By - dt*curly
    Bz = Bz - dt*curlz
    # update the magnetic field from Maxwell's equations

    return (Bx, By, Bz)

@partial(jit, static_argnums=(1))
def digital_filter(phi, alpha):
    """
    Apply a digital filter to the electric potential array.

    Args:
        phi (ndarray): Electric potential array.
        alpha (float): Filter coefficient.

    Returns:
        ndarray: Filtered electric potential array.
    """
    filter_phi = alpha * phi +  (  (1 - alpha) / 6 ) * (  jnp.roll(phi, shift=1, axis=0) + jnp.roll(phi, shift=-1, axis=0) + \
                                                            jnp.roll(phi, shift=1, axis=1) + jnp.roll(phi, shift=-1, axis=1) + \
                                                                jnp.roll(phi, shift=1, axis=2) + jnp.roll(phi, shift=-1, axis=2) )
    return filter_phi