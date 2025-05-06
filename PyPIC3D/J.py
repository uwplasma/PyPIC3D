import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
# import external libraries

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

    return (Jx, Jy, Jz)
    # return the current corrections