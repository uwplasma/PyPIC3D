import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.rho import compute_rho

@jit
def J_from_rhov(particles, J, constants, world, grid):
    """
    Compute the current density from the charge density and particle velocities.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        rho (ndarray): Charge density array.
        J (tuple): Current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
        constants (dict): Dictionary containing physical constants.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
    """

    C = constants['C']
    # speed of light

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the world parameters

    Jx, Jy, Jz = J
    # unpack the values of J
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0
    J = (Jx, Jy, Jz)
    # initialize the current density as a tuple

    for species in particles:
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        # get the number of particles and their charge
        particle_x, particle_y, particle_z = species.get_position()
        # get the position of the particles in the species

        vx, vy, vz = species.get_velocity()
        # get the velocity of the particles in the species

        shape_factor = species.get_shape()

        def add_to_J(particle, J):
            x = particle_x.at[particle].get()
            y = particle_y.at[particle].get()
            z = particle_z.at[particle].get()
            # get the position of the particle
            vx_particle = vx.at[particle].get()
            vy_particle = vy.at[particle].get()
            vz_particle = vz.at[particle].get()
            # get the velocity of the particle

            J = lax.cond(
                shape_factor == 1,
                lambda _: J_first_order_weighting(charge, x, y, z, vx_particle, vy_particle, vz_particle, J, dx, dy, dz,  grid),
                lambda _: J_second_order_weighting(charge, x, y, z, vx_particle, vy_particle, vz_particle, J, dx, dy, dz, grid),
                operand=None
            )
            # use first-order weighting to distribute the current density

            return J
        # add the particle species to the charge density array
        J = jax.lax.fori_loop(0, N_particles, add_to_J, J)
    # loop over all particles in the species and add their contribution to the current density

    return J


@jit
def J_first_order_weighting(q, x, y, z, vx, vy, vz, J, dx, dy, dz, grid):

    Jx, Jy, Jz = J
    # unpack the values of J

    Nx, Ny, Nz = Jx.shape
    # get the shape of the charge density array

    x0 = jnp.floor((x - grid[0][0]) / dx).astype(int)
    y0 = jnp.floor((y - grid[1][0]) / dy).astype(int)
    z0 = jnp.floor((z - grid[2][0]) / dz).astype(int)
    # Calculate the nearest grid points

    deltax = x - jnp.floor(x / dx) * dx
    deltay = y - jnp.floor(y / dy) * dy
    deltaz = z - jnp.floor(z / dz) * dz
    # Calculate the difference between the particle position and the nearest grid point


    x1 = wrap_around(x0 + 1, Nx)
    y1 = wrap_around(y0 + 1, Ny)
    z1 = wrap_around(z0 + 1, Nz)
    # Calculate the index of the next grid point

    dv = dx * dy * dz
    # Calculate the volume of each grid point

    ##################################### WEIGHTING FACTORS ##########################################################
    Sx0 = 1 - (deltax/dx)
    Sy0 = 1 - (deltay/dy)
    Sz0 = 1 - (deltaz/dz)
    # calculate the weight for the center grid points
    Sx1 = deltax/dx
    Sy1 = deltay/dy
    Sz1 = deltaz/dz
    # calculate the weight for the next grid points
    ####################################################################################################################

    ####################################### JX DISTRIBUTION ########################################################
    Jx = Jx.at[x0, y0, z0].add((q * vx / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z0].add((q * vx / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x0, y1, z0].add((q * vx / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x0, y0, z1].add((q * vx / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z0].add((q * vx / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z1].add((q * vx / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x0, y1, z1].add((q * vx / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z1].add((q * vx / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    # Distribute the current density in the x direction to the grid points
    #####################################################################################################################

    ####################################### JY DISTRIBUTION ########################################################
    Jy = Jy.at[x0, y0, z0].add((q * vy / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z0].add((q * vy / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x0, y1, z0].add((q * vy / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x0, y0, z1].add((q * vy / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z0].add((q * vy / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z1].add((q * vy / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x0, y1, z1].add((q * vy / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z1].add((q * vy / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    # Distribute the current density in the y direction to the grid points
    #####################################################################################################################

    ####################################### JZ DISTRIBUTION ########################################################
    Jz = Jz.at[x0, y0, z0].add((q * vz / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z0].add((q * vz / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x0, y1, z0].add((q * vz / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x0, y0, z1].add((q * vz / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z0].add((q * vz / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z1].add((q * vz / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x0, y1, z1].add((q * vz / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z1].add((q * vz / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    # Distribute the current density in the z direction to the grid points
    #####################################################################################################################


    return (Jx, Jy, Jz)


@jit
def J_second_order_weighting(q, x, y, z, vx, vy, vz, J, dx, dy, dz, grid):
    """
    Distribute the current of a particle to the surrounding grid points using second-order weighting.

    Args:
        q (float): Charge of the particle.
        x, y, z (float): Position of the particle.
        vx, vy, vz (float): Velocity components of the particle.
        J (tuple): Current density arrays (Jx, Jy, Jz).
        rho (ndarray): Charge density array (for shape).
        dx, dy, dz (float): Grid spacing.
        x_wind, y_wind, z_wind (float): Window offsets.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz).
    """

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape

    x0 = jnp.floor((x - grid[0][0]) / dx).astype(int)
    y0 = jnp.floor((y - grid[1][0]) / dy).astype(int)
    z0 = jnp.floor((z - grid[2][0]) / dz).astype(int)
    # Calculate the nearest grid points

    deltax = x - jnp.floor( x / dx) * dx
    deltay = y - jnp.floor( y / dy) * dy
    deltaz = z - jnp.floor( z / dz) * dz
    # Calculate the difference between the particle position and the nearest grid point

    x1 = wrap_around(x0 + 1, Nx)
    y1 = wrap_around(y0 + 1, Ny)
    z1 = wrap_around(z0 + 1, Nz)

    x_minus1 = x0 - 1
    y_minus1 = y0 - 1
    z_minus1 = z0 - 1

    dv = dx * dy * dz

    # Weighting factors
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

    Sx1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) - (deltax/dx))**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )
    Sy1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) - (deltay/dy))**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) - (deltaz/dz))**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )

    Sx_minus1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) + (deltax/dx))**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )
    Sy_minus1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) + (deltay/dy))**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz_minus1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) + (deltaz/dz))**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )

    # Jx distribution
    Jx = Jx.at[x0, y0, z0].add((q * vx / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z0].add((q * vx / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x0, y1, z0].add((q * vx / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x0, y0, z1].add((q * vx / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z0].add((q * vx / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z1].add((q * vx / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x0, y1, z1].add((q * vx / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z1].add((q * vx / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jx = Jx.at[x_minus1, y0, z0].add((q * vx / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x0, y_minus1, z0].add((q * vx / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jx = Jx.at[x0, y0, z_minus1].add((q * vx / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jx = Jx.at[x_minus1, y_minus1, z0].add((q * vx / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jx = Jx.at[x_minus1, y0, z_minus1].add((q * vx / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jx = Jx.at[x0, y_minus1, z_minus1].add((q * vx / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jx = Jx.at[x_minus1, y_minus1, z_minus1].add((q * vx / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    # Jy distribution
    Jy = Jy.at[x0, y0, z0].add((q * vy / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z0].add((q * vy / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x0, y1, z0].add((q * vy / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x0, y0, z1].add((q * vy / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z0].add((q * vy / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z1].add((q * vy / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x0, y1, z1].add((q * vy / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z1].add((q * vy / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jy = Jy.at[x_minus1, y0, z0].add((q * vy / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x0, y_minus1, z0].add((q * vy / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jy = Jy.at[x0, y0, z_minus1].add((q * vy / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jy = Jy.at[x_minus1, y_minus1, z0].add((q * vy / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jy = Jy.at[x_minus1, y0, z_minus1].add((q * vy / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jy = Jy.at[x0, y_minus1, z_minus1].add((q * vy / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jy = Jy.at[x_minus1, y_minus1, z_minus1].add((q * vy / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    # Jz distribution
    Jz = Jz.at[x0, y0, z0].add((q * vz / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z0].add((q * vz / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x0, y1, z0].add((q * vz / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x0, y0, z1].add((q * vz / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z0].add((q * vz / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z1].add((q * vz / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x0, y1, z1].add((q * vz / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z1].add((q * vz / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jz = Jz.at[x_minus1, y0, z0].add((q * vz / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x0, y_minus1, z0].add((q * vz / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jz = Jz.at[x0, y0, z_minus1].add((q * vz / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jz = Jz.at[x_minus1, y_minus1, z0].add((q * vz / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jz = Jz.at[x_minus1, y0, z_minus1].add((q * vz / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jz = Jz.at[x0, y_minus1, z_minus1].add((q * vz / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jz = Jz.at[x_minus1, y_minus1, z_minus1].add((q * vz / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    return (Jx, Jy, Jz)

@jit
def wrap_around(ix, size):
    """Wrap around index (scalar or 1D array) to ensure it is within bounds."""
    return jnp.where(ix > size - 1, ix - size, ix)

@jit
def VB_current(particles, J, constants, world, grid):
    """
    Apply Villasenor-Buneman correction to ensure rigorous charge conservation for local electromagnetic field solvers.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        Nz (int): Number of grid points in the z-direction.

    Returns:
        tuple: Corrected current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.

    References:
        Villasenor, J., & Buneman, O. (1992). Rigorous charge conservation for local electromagnetic field solvers.
        Computer Physics Communications, 69(2-3), 306-316.
    """

    Jx, Jy, Jz = J
    # unpack the values of J

    C = constants['C']
    # speed of light

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the world parameters

    for species in particles:
        q = species.get_charge()
        # get the charge of the species
        old_x, old_y, old_z = species.get_old_position()
        # get the old position of the particles in the species
        x, y, z = species.get_position()
        # get the position of the particles in the species

        zeta1 = jnp.floor((x + x_wind / 2) / dx).astype(int)
        eta1 = jnp.floor((y + y_wind / 2) / dy).astype(int)
        xi1 = jnp.floor((z + z_wind / 2) / dz).astype(int)
        zeta2 = jnp.floor((old_x + x_wind / 2) / dx).astype(int)
        eta2 = jnp.floor((old_y + y_wind / 2) / dy).astype(int)
        xi2 = jnp.floor((old_z + z_wind / 2) / dz).astype(int)
        # Calculate the nearest grid points

        dx, dy, dz = species.get_resolution()
        # get the resolution of the species

        deltax = zeta2 - zeta1
        deltay = eta2 - eta1
        deltaz = xi2 - xi1

        zetabar = 0.5*(zeta1 + zeta2)
        etabar = 0.5*(eta1 + eta2)
        xibar = 0.5*(xi1 + xi2)
        # compute the displacement variables from Buneman, Vilasenor 1991

        ix, iy, iz = species.get_index()
        # get the index of the species

        dq = q/dx/dy/dz
        # charge differential

        ix1 = wrap_around(ix + 1, Jx.shape[0])
        iy1 = wrap_around(iy + 1, Jx.shape[1])
        iz1 = wrap_around(iz + 1, Jx.shape[2])

        Jx = Jx.at[ix, iy1, iz1].add( dq*C*(deltax*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first x correction for charge conservation along i+1/2, j, k
        Jx = Jx.at[ix, iy, iz1].add( dq*C*(deltax*(1-etabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second x correction for charge conservation along i+1/2, j, k+1
        Jx = Jx.at[ix, iy1, iz].add( dq*C*(deltax*etabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third x correction for charge conservation along i+1/2, j+1, k
        Jx = Jx.at[ix, iy, iz].add( dq*C*(deltax*(1-etabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth x correction for charge conservation along i+1/2, j, k

        Jy = Jy.at[ix1, iy, iz1].add( dq*C*(deltay*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first y correction for charge conservation along i, j+1/2, k
        Jy = Jy.at[ix1, iy, iz].add( dq*C*(deltay*(1-zetabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second y correction for charge conservation along i+1, j+1/2, k
        Jy = Jy.at[ix, iy, iz1].add( dq*C*(deltay*zetabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third y correction for charge conservation along i, j+1/2, k+1
        Jy = Jy.at[ix, iy, iz].add( dq*C*(deltay*(1-zetabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth y correction for charge conservation along i, j+1/2, k

        Jz = Jz.at[ix1, iy1, iz].add(dq*C*(deltaz*(1-zetabar)*etabar + deltax*deltay*deltaz/12))
        # compute the first z correction for charge conservation along i+1, j+1, k+1/2
        Jz = Jz.at[ix, iy1, iz].add(dq*C*(deltaz*(1-zetabar)*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the second z correction for charge conservation along i, j+1, k+1/2
        Jz = Jz.at[ix1, iy, iz].add(dq*C*(deltaz*zetabar*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the third z correction for charge conservation along i+1, j, k+1/2
        Jz = Jz.at[ix, iy, iz].add(dq*C*(deltaz*(1-zetabar)*(1-etabar) + deltax*deltay*deltaz/12))
        # compute the fourth z correction for charge conservation along i, j, k+1/2

    return (Jx, Jy, Jz)
    # return the current corrections

def Esirkepov_current(particles, J, constants, world, grid):

    Jx, Jy, Jz = J
    # unpack the values of J

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid spacing parameters
    dt = world['dt']
    # get the time step
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']
    # get the number of grid points in each direction


    for species in particles:
        q = species.get_charge()
        # get the charge of the species

        old_x, old_y, old_z = species.get_old_position()
        # get the old position of the particles in the species
        x, y, z = species.get_position()
        # get the position of the particles in the species

        shape_factor = species.get_shape()
        # get the shape factor of the species

        x0 = jnp.floor((x - grid[0][0]) / dx).astype(int)
        y0 = jnp.floor((y - grid[1][0]) / dy).astype(int)
        z0 = jnp.floor((z - grid[2][0]) / dz).astype(int)
        # Calculate the nearest grid points

        x1 = wrap_around(x0 + 1, Nx)
        y1 = wrap_around(y0 + 1, Ny)
        z1 = wrap_around(z0 + 1, Nz)
        # Calculate the index of the next grid point

        x_minus1 = wrap_around(x1 - 1, Nx)
        y_minus1 = wrap_around(y1 - 1, Ny)
        z_minus1 = wrap_around(z1 - 1, Nz)
        # Calculate the index of the next grid point

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]
        # List of grid points in each direction

        old_deltax = old_x - jnp.floor(old_x / dx) * dx
        old_deltay = old_y - jnp.floor(old_y / dy) * dy
        old_deltaz = old_z - jnp.floor(old_z / dz) * dz
        # Calculate the difference between the particle position and the nearest grid point
        new_deltax = x - jnp.floor(x / dx) * dx
        new_deltay = y - jnp.floor(y / dy) * dy
        new_deltaz = z - jnp.floor(z / dz) * dz
        # Calculate the difference between the particle position and the nearest grid point

        old_x_weights, old_y_weights, old_z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None
        )
        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(new_deltax, new_deltay, new_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(new_deltax, new_deltay, new_deltaz, dx, dy, dz),
            operand=None
        )
        # Calculate the weights for the grid points

        d_Sx = []
        d_Sy = []
        d_Sz = []

        for i in range(len(x_weights)):
            d_Sx.append(x_weights[i] - old_x_weights[i])
            d_Sy.append(y_weights[i] - old_y_weights[i])
            d_Sz.append(z_weights[i] - old_z_weights[i])
        # Calculate the difference in weights for the central grid points

        Wx = []
        Wy = []
        Wz = []

        for i in range(len(x_weights)):
            Wx.append( d_Sx[i] * ( old_y_weights[i] * old_z_weights[i] + 0.5 * d_Sy[i] * old_z_weights[i] \
               + 0.5 * old_y_weights[i] * d_Sz[i] + d_Sy[i] * d_Sz[i] / 3))
            Wy.append( d_Sy[i] * ( old_x_weights[i] * old_z_weights[i] + 0.5 * d_Sx[i] * old_z_weights[i] \
               + 0.5 * old_x_weights[i] * d_Sz[i] + d_Sx[i] * d_Sz[i] / 3))
            Wz.append( d_Sz[i] * ( old_x_weights[i] * old_y_weights[i] + 0.5 * d_Sx[i] * old_y_weights[i] \
               + 0.5 * old_x_weights[i] * d_Sy[i] + d_Sx[i] * d_Sy[i] / 3))


        for i in range(len(xpts)):
            x = xpts[i]
            y = ypts[i]
            z = zpts[i]
            # Loop over the grid points in each direction
            Jx = Jx.at[x, y, z].add( Wx[i] *  (q / dy / dz / dt), mode='drop')
            Jy = Jy.at[x, y, z].add( Wy[i] *  (q / dx / dz / dt), mode='drop')
            Jz = Jz.at[x, y, z].add( Wz[i] *  (q / dx / dy / dt), mode='drop')
            # Distribute the current density to the grid points

    return (Jx, Jy, Jz)

def get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the second-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

    Sx1 = jnp.where(deltax <= dx/2, (1/2) * ((1/2) - (deltax/dx))**2, 0.0)
    Sy1 = jnp.where(deltay <= dy/2, (1/2) * ((1/2) - (deltay/dy))**2, 0.0)
    Sz1 = jnp.where(deltaz <= dz/2, (1/2) * ((1/2) - (deltaz/dz))**2, 0.0)

    Sx_minus1 = jnp.where(deltax <= dx/2, (1/2) * ((1/2) + (deltax/dx))**2, 0.0)
    Sy_minus1 = jnp.where(deltay <= dy/2, (1/2) * ((1/2) + (deltay/dy))**2, 0.0)
    Sz_minus1 = jnp.where(deltaz <= dz/2, (1/2) * ((1/2) + (deltaz/dz))**2, 0.0)

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights


def get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the first-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = jnp.asarray(1 - deltax / dx)
    Sy0 = jnp.asarray(1 - deltay / dy)
    Sz0 = jnp.asarray(1 - deltaz / dz)

    Sx1 = jnp.asarray(deltax / dx)
    Sy1 = jnp.asarray(deltay / dy)
    Sz1 = jnp.asarray(deltaz / dz)

    Sx_minus1 = jnp.zeros_like(Sx0)
    Sy_minus1 = jnp.zeros_like(Sy0)
    Sz_minus1 = jnp.zeros_like(Sz0)
    # No second-order weights for first-order weighting

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights