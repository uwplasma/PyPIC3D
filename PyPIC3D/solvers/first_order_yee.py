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



@partial(jit, static_argnames=("curl_func", "x_bc", "y_bc", "z_bc"))
def update_E(E, B, J, world, constants, curl_func, x_bc, y_bc, z_bc):
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
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the time resolution and grid spacings
    C = constants['C']
    eps = constants['eps']
    # get the time resolution and necessary constants

    Bx = jnp.pad(Bx, ((1,1), (1,1), (1,1)), mode="wrap")
    By = jnp.pad(By, ((1,1), (1,1), (1,1)), mode="wrap")
    Bz = jnp.pad(Bz, ((1,1), (1,1), (1,1)), mode="wrap")
    # pad the magnetic field components for periodic boundary conditions

    dBz_dy = (jnp.roll(Bz, shift=-1, axis=1) - Bz) / dy
    dBx_dy = (jnp.roll(Bx, shift=-1, axis=1) - Bx) / dy
    dBy_dz = (jnp.roll(By, shift=-1, axis=2) - By) / dz
    dBx_dz = (jnp.roll(Bx, shift=-1, axis=2) - Bx) / dz
    dBz_dx = (jnp.roll(Bz, shift=-1, axis=0) - Bz) / dx
    dBy_dx = (jnp.roll(By, shift=-1, axis=0) - By) / dx

    curl_x = (dBz_dy - dBy_dz)[1:-1,1:-1,1:-1]
    curl_y = (dBx_dz - dBz_dx)[1:-1,1:-1,1:-1]
    curl_z = (dBy_dx - dBx_dy)[1:-1,1:-1,1:-1]
    # calculate the curl of the magnetic field

    Ex = Ex + ( C**2 * curl_x - Jx / eps ) * dt
    Ey = Ey + ( C**2 * curl_y - Jy / eps ) * dt
    Ez = Ez + ( C**2 * curl_z - Jz / eps ) * dt
    # update the electric field from Maxwell's equations

    ### Conducting Boundaries ####
    
    ### X BC ###
    Ey = lax.cond(
        x_bc == 'conducting',
        lambda Ey: Ey.at[0,:,:].set(0.0).at[-1,:,:].set(0.0).at[-2,:,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ey: Ey,
        operand=Ey
    )

    Ez = lax.cond(
        x_bc == 'conducting',
        lambda Ez: Ez.at[0,:,:].set(0.0).at[-1,:,:].set(0.0).at[-2,:,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ez: Ez,
        operand=Ez
    )
    # set Ey and Ez to 0 at the x boundaries for conducting boundaries

    ### Y BC ###
    Ex = lax.cond(
        y_bc == 'conducting',
        lambda Ex: Ex.at[:,0,:].set(0.0).at[:,-1,:].set(0.0).at[:,-2,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ex: Ex,
        operand=Ex
    )

    Ez = lax.cond(
        y_bc == 'conducting',
        lambda Ez: Ez.at[:,0,:].set(0.0).at[:,-1,:].set(0.0).at[:,-2,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ez: Ez,
        operand=Ez
    )
    # set Ex and Ez to 0 at the y boundaries for conducting boundaries

    ### Z BC ###
    Ex = lax.cond(
        z_bc == 'conducting',
        lambda Ex: Ex.at[:,:,0].set(0.0).at[:,:,-1].set(0.0).at[:,:,-2].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ex: Ex,
        operand=Ex
    )

    Ey = lax.cond(
        z_bc == 'conducting',
        lambda Ey: Ey.at[:,:,0].set(0.0).at[:,:,-1].set(0.0).at[:,:,-2].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ey: Ey,
        operand=Ey
    )
    # set Ex and Ey to 0 at the z boundaries for conducting boundaries

    alpha = constants['alpha']
    Ex = digital_filter(Ex, alpha)
    Ey = digital_filter(Ey, alpha)
    Ez = digital_filter(Ez, alpha)
    # apply a digital filter to the electric field components

    return (Ex, Ey, Ez)


@partial(jit, static_argnames=("curl_func", "x_bc", "y_bc", "z_bc"))
def update_B(E, B, world, constants, curl_func, x_bc, y_bc, z_bc):
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
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the grid spacings

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # unpack the E and B fields

    Ex = jnp.pad(Ex, ((1,1), (1,1), (1,1)), mode="wrap")
    Ey = jnp.pad(Ey, ((1,1), (1,1), (1,1)), mode="wrap")
    Ez = jnp.pad(Ez, ((1,1), (1,1), (1,1)), mode="wrap")
    # pad the electric field components for periodic boundary conditions

    dEz_dy = (Ez - jnp.roll(Ez, shift=1, axis=1)) / dy
    dEx_dy = (Ex - jnp.roll(Ex, shift=1, axis=1)) / dy
    dEy_dz = (Ey - jnp.roll(Ey, shift=1, axis=2)) / dz
    dEx_dz = (Ex - jnp.roll(Ex, shift=1, axis=2)) / dz
    dEz_dx = (Ez - jnp.roll(Ez, shift=1, axis=0)) / dx
    dEy_dx = (Ey - jnp.roll(Ey, shift=1, axis=0)) / dx

    curl_x = (dEz_dy - dEy_dz)[1:-1,1:-1,1:-1]
    curl_y = (dEx_dz - dEz_dx)[1:-1,1:-1,1:-1]
    curl_z = (dEy_dx - dEx_dy)[1:-1,1:-1,1:-1]
    # calculate the curl of the electric field

    Bx = Bx - dt*curl_x
    By = By - dt*curl_y
    Bz = Bz - dt*curl_z
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