import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
# import external libraries

from PyPIC3D.rho import compute_rho

@jit
def J_from_rhov(particles, J, rho, constants, world):
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

    new_rho = compute_rho(particles, rho, world)
    old_rho = rho
    # compute the charge density from the particles

    rho = 0.5 * (new_rho + old_rho)
    # average the charge density to ensure stability

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

        def add_to_J(particle, J):
            x = particle_x.at[particle].get()
            y = particle_y.at[particle].get()
            z = particle_z.at[particle].get()
            # get the position of the particle
            vx_particle = vx.at[particle].get()
            vy_particle = vy.at[particle].get()
            vz_particle = vz.at[particle].get()
            # get the velocity of the particle

            J = J_first_order_weighting(
                charge, x, y, z, vx_particle, vy_particle, vz_particle, J, rho,
                dx, dy, dz, x_wind, y_wind, z_wind
            )
            # use first-order weighting to distribute the current density

            return J
        # add the particle species to the charge density array
        J = jax.lax.fori_loop(0, N_particles, add_to_J, J)
        # iterate over each particle in the species and update the current density
        # using the J_first_order_weighting function

    return J, new_rho


@jit
def J_first_order_weighting(q, x, y, z, vx, vy, vz, J, rho, dx, dy, dz, x_wind, y_wind, z_wind):

    Jx, Jy, Jz = J
    # unpack the values of J

    Nx, Ny, Nz = rho.shape
    # get the shape of the charge density array

    x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
    y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
    z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)
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
def wrap_around(ix, size):
    """Wrap around index (scalar or 1D array) to ensure it is within bounds."""
    return jnp.where(ix > size - 1, ix - size, ix)

#@partial(jit, static_argnums=(1, 2, 3))
def VB_correction(particles, J, constants):
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

    for species in particles:
        q = species.get_charge()
        # get the charge of the species
        zeta1, zeta2, eta1, eta2, xi1, xi2 = species.get_subcell_position()
        # get the particle positions

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