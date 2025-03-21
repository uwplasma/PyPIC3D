import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
# import external libraries


@partial(jit, static_argnums=(5))
def compute_current_density(particles, Jx, Jy, Jz, world, GPUs):
    """
    Computes the current density for a given set of particles in a simulation world.

    Args:
        particles (list): A list of particle species, each containing methods to get the number of particles,
                        their positions, velocities, and charge.
        Jx, Jy, Jz (numpy.ndarray): The current density arrays to be updated.
        world (dict): A dictionary containing the simulation world parameters such as grid spacing (dx, dy, dz)
                    and window dimensions (x_wind, y_wind, z_wind).
        GPUs (bool): A flag indicating whether to use GPU acceleration for the computation.

    Returns:
        tuple: The updated current density arrays (Jx, Jy, Jz).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']

    Jx = jnp.zeros_like(Jx)
    Jy = jnp.zeros_like(Jy)
    Jz = jnp.zeros_like(Jz)

    for species in particles:
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        if N_particles > 0:
            particle_x, particle_y, particle_z = species.get_position()
            particle_vx, particle_vy, particle_vz = species.get_velocity()
            Jx, Jy, Jz = update_current_density(N_particles, particle_x, particle_y, particle_z, particle_vx, particle_vy, particle_vz, dx, dy, dz, charge, x_wind, y_wind, z_wind, Jx, Jy, Jz, GPUs)
    return Jx, Jy, Jz

@jit
def update_current_density(Nparticles, particlex, particley, particlez, particlevx, particlevy, particlevz, dx, dy, dz, q, x_wind, y_wind, z_wind, Jx, Jy, Jz, GPUs=False):

    def addto_J(particle, J):
        Jx, Jy, Jz = J
        x = particlex.at[particle].get()
        y = particley.at[particle].get()
        z = particlez.at[particle].get()
        vx = particlevx.at[particle].get()
        vy = particlevy.at[particle].get()
        vz = particlevz.at[particle].get()

        # Calculate the nearest grid points
        x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
        y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
        z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)

        # Calculate the difference between the particle position and the nearest grid point
        deltax = (x + x_wind / 2) - x0 * dx
        deltay = (y + y_wind / 2) - y0 * dy
        deltaz = (z + z_wind / 2) - z0 * dz

        # Calculate the index of the next grid point
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Calculate the weights for the surrounding grid points
        wx = deltax / dx
        wy = deltay / dy
        wz = deltaz / dz

        # Calculate the volume of each grid point
        dv = dx * dy * dz

        # Distribute the current density to the surrounding grid points
        Jx = Jx.at[x0, y0, z0].add(q * vx * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jx = Jx.at[x1, y0, z0].add(q * vx * wx * (1 - wy) * (1 - wz) / dv)
        Jx = Jx.at[x0, y1, z0].add(q * vx * (1 - wx) * wy * (1 - wz) / dv)
        Jx = Jx.at[x0, y0, z1].add(q * vx * (1 - wx) * (1 - wy) * wz / dv)
        Jx = Jx.at[x1, y1, z0].add(q * vx * wx * wy * (1 - wz) / dv)
        Jx = Jx.at[x1, y0, z1].add(q * vx * wx * (1 - wy) * wz / dv)
        Jx = Jx.at[x0, y1, z1].add(q * vx * (1 - wx) * wy * wz / dv)
        Jx = Jx.at[x1, y1, z1].add(q * vx * wx * wy * wz / dv)

        Jy = Jy.at[x0, y0, z0].add(q * vy * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jy = Jy.at[x1, y0, z0].add(q * vy * wx * (1 - wy) * (1 - wz) / dv)
        Jy = Jy.at[x0, y1, z0].add(q * vy * (1 - wx) * wy * (1 - wz) / dv)
        Jy = Jy.at[x0, y0, z1].add(q * vy * (1 - wx) * (1 - wy) * wz / dv)
        Jy = Jy.at[x1, y1, z0].add(q * vy * wx * wy * (1 - wz) / dv)
        Jy = Jy.at[x1, y0, z1].add(q * vy * wx * (1 - wy) * wz / dv)
        Jy = Jy.at[x0, y1, z1].add(q * vy * (1 - wx) * wy * wz / dv)
        Jy = Jy.at[x1, y1, z1].add(q * vy * wx * wy * wz / dv)

        Jz = Jz.at[x0, y0, z0].add(q * vz * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jz = Jz.at[x1, y0, z0].add(q * vz * wx * (1 - wy) * (1 - wz) / dv)
        Jz = Jz.at[x0, y1, z0].add(q * vz * (1 - wx) * wy * (1 - wz) / dv)
        Jz = Jz.at[x0, y0, z1].add(q * vz * (1 - wx) * (1 - wy) * wz / dv)
        Jz = Jz.at[x1, y1, z0].add(q * vz * wx * wy * (1 - wz) / dv)
        Jz = Jz.at[x1, y0, z1].add(q * vz * wx * (1 - wy) * wz / dv)
        Jz = Jz.at[x0, y1, z1].add(q * vz * (1 - wx) * wy * wz / dv)
        Jz = Jz.at[x1, y1, z1].add(q * vz * wx * wy * wz / dv)

        return Jx, Jy, Jz

    return jax.lax.fori_loop(0, Nparticles-1, addto_J, (Jx, Jy, Jz))

#@partial(jit, static_argnums=(1, 2, 3))
def VB_correction(particles, Jx, Jy, Jz, constants):
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

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0

    C = constants['C']
    # speed of light

    for species in particles:
        q = species.get_charge()
        # get the charge of the species
        zeta1, zeta2, eta1, eta2, xi1, xi2 = species.get_subcell_position()
        # get the particle positions

        dx, dy, dz = species.get_resolution()
        # get the resolution of the species

        vx, vy, vz = species.get_velocity()
        # get the velocity of the species

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

        Jx = Jx.at[ix, iy+1, iz+1].add( dq*C*(deltax*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first x correction for charge conservation along i+1/2, j, k
        Jx = Jx.at[ix, iy, iz+1].add( dq*C*(deltax*(1-etabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second x correction for charge conservation along i+1/2, j, k+1
        Jx = Jx.at[ix, iy+1, iz].add( dq*C*(deltax*etabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third x correction for charge conservation along i+1/2, j+1, k
        Jx = Jx.at[ix, iy, iz].add( dq*C*(deltax*(1-etabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth x correction for charge conservation along i+1/2, j, k

        Jy = Jy.at[ix+1, iy, iz+1].add( dq*C*(deltay*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first y correction for charge conservation along i, j+1/2, k
        Jy = Jy.at[ix+1, iy, iz].add( dq*C*(deltay*(1-zetabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second y correction for charge conservation along i+1, j+1/2, k
        Jy = Jy.at[ix, iy, iz+1].add( dq*C*(deltay*zetabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third y correction for charge conservation along i, j+1/2, k+1
        Jy = Jy.at[ix, iy, iz].add( dq*C*(deltay*(1-zetabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth y correction for charge conservation along i, j+1/2, k

        Jz = Jz.at[ix+1, iy+1, iz].add(dq*C*(deltaz*(1-zetabar)*etabar + deltax*deltay*deltaz/12))
        # compute the first z correction for charge conservation along i+1, j+1, k+1/2
        Jz = Jz.at[ix, iy+1, iz].add(dq*C*(deltaz*(1-zetabar)*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the second z correction for charge conservation along i, j+1, k+1/2
        Jz = Jz.at[ix+1, iy, iz].add(dq*C*(deltaz*zetabar*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the third z correction for charge conservation along i+1, j, k+1/2
        Jz = Jz.at[ix, iy, iz].add(dq*C*(deltaz*(1-zetabar)*(1-etabar) + deltax*deltay*deltaz/12))
        # compute the fourth z correction for charge conservation along i, j, k+1/2

    return Jx, Jy, Jz
    # return the current corrections


# def EZ_current_deposition(particles, Jx, Jy, Jz, world):
