import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
# import external libraries

from PyPIC3D.utils import digital_filter
# import internal libraries

BC_PERIODIC = 0
BC_CONDUCTING = 1


def _get_field_bc_code(world, axis):
    """
    Get the boundary condition code for an axis, supporting legacy and refactored world schemas.
    """
    if 'boundary_conditions' in world and axis in world['boundary_conditions']:
        bc = world['boundary_conditions'][axis]
    else:
        legacy_key = f"{axis}_bc"
        bc = world[legacy_key] if legacy_key in world else BC_PERIODIC

    if isinstance(bc, str):
        return BC_CONDUCTING if bc == "conducting" else BC_PERIODIC
    return bc


@partial(jit, static_argnames=("curl_func",))
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
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    x_bc = _get_field_bc_code(world, 'x')
    y_bc = _get_field_bc_code(world, 'y')
    z_bc = _get_field_bc_code(world, 'z')
    # get the time resolution and grid spacings
    C = constants['C']
    eps = constants['eps']
    # get the time resolution and necessary constants

    dBz_dy = (jnp.roll(Bz, shift=-1, axis=1) - Bz) / dy
    dBx_dy = (jnp.roll(Bx, shift=-1, axis=1) - Bx) / dy
    dBy_dz = (jnp.roll(By, shift=-1, axis=2) - By) / dz
    dBx_dz = (jnp.roll(Bx, shift=-1, axis=2) - Bx) / dz
    dBz_dx = (jnp.roll(Bz, shift=-1, axis=0) - Bz) / dx
    dBy_dx = (jnp.roll(By, shift=-1, axis=0) - By) / dx

    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy
    # calculate the curl of the magnetic field

    Ex = Ex + ( C**2 * curl_x - Jx / eps ) * dt
    Ey = Ey + ( C**2 * curl_y - Jy / eps ) * dt
    Ez = Ez + ( C**2 * curl_z - Jz / eps ) * dt
    # update the electric field from Maxwell's equations

    ### Conducting Boundaries ####
    
    ### X BC ###
    Ey = lax.cond(
        x_bc == BC_CONDUCTING,
        lambda Ey: Ey.at[0,:,:].set(0.0).at[-1,:,:].set(0.0).at[-2,:,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ey: Ey,
        operand=Ey
    )

    Ez = lax.cond(
        x_bc == BC_CONDUCTING,
        lambda Ez: Ez.at[0,:,:].set(0.0).at[-1,:,:].set(0.0).at[-2,:,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ez: Ez,
        operand=Ez
    )
    # set Ey and Ez to 0 at the x boundaries for conducting boundaries

    ### Y BC ###
    Ex = lax.cond(
        y_bc == BC_CONDUCTING,
        lambda Ex: Ex.at[:,0,:].set(0.0).at[:,-1,:].set(0.0).at[:,-2,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ex: Ex,
        operand=Ex
    )

    Ez = lax.cond(
        y_bc == BC_CONDUCTING,
        lambda Ez: Ez.at[:,0,:].set(0.0).at[:,-1,:].set(0.0).at[:,-2,:].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ez: Ez,
        operand=Ez
    )
    # set Ex and Ez to 0 at the y boundaries for conducting boundaries

    ### Z BC ###
    Ex = lax.cond(
        z_bc == BC_CONDUCTING,
        lambda Ex: Ex.at[:,:,0].set(0.0).at[:,:,-1].set(0.0).at[:,:,-2].set(0.0),
        # the left inner cell next to the boundary to 0,
        # and the right 2 inner cells next to the boundary to 0
        lambda Ex: Ex,
        operand=Ex
    )

    Ey = lax.cond(
        z_bc == BC_CONDUCTING,
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


@partial(jit, static_argnames=("curl_func",))
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
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the grid spacings

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # unpack the E and B fields

    dEz_dy = (Ez - jnp.roll(Ez, shift=1, axis=1)) / dy
    dEx_dy = (Ex - jnp.roll(Ex, shift=1, axis=1)) / dy
    dEy_dz = (Ey - jnp.roll(Ey, shift=1, axis=2)) / dz
    dEx_dz = (Ex - jnp.roll(Ex, shift=1, axis=2)) / dz
    dEz_dx = (Ez - jnp.roll(Ez, shift=1, axis=0)) / dx
    dEy_dx = (Ey - jnp.roll(Ey, shift=1, axis=0)) / dx

    curl_x = dEz_dy - dEy_dz
    curl_y = dEx_dz - dEz_dx
    curl_z = dEy_dx - dEx_dy
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
