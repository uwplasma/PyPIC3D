import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

from PyPIC3D.spectral import spectral_divergence, spectral_gradient
from PyPIC3D.fdtd import centered_finite_difference_divergence, centered_finite_difference_gradient

def marder_correction(Ex, Ey, Ez, rho, world, eps, dt, solver='spectral'):
    """
    Apply Marder's correction to the electric field components using different solvers.

    Parameters:
    Ex (ndarray): Electric field component in the x-direction.
    Ey (ndarray): Electric field component in the y-direction.
    Ez (ndarray): Electric field component in the z-direction.
    rho (ndarray): Charge density.
    world (dict): Dictionary containing the grid spacing with keys 'dx', 'dy', and 'dz'.
    eps (float): Permittivity of the medium.
    solver (str): The solver to use for computing the divergence and gradient ('spectral' or 'finite_difference').

    Returns:
    tuple: A tuple containing the correction components (Ex, Ey, Ez).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    if solver == 'spectral':
        div_E = spectral_divergence(Ex, Ey, Ez, dx, dy, dz)
        F = div_E - rho/eps
        dF = spectral_gradient(F, dx, dy, dz)
    elif solver == 'fdtd':
        div_E = centered_finite_difference_divergence(Ex, Ey, Ez, dx, dy, dz)
        F = div_E - rho/eps
        dF = centered_finite_difference_gradient(F, dx, dy, dz)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    Ex = Ex - dF[0]*dt/2
    Ey = Ey - dF[1]*dt/2
    Ez = Ez - dF[2]*dt/2

    return Ex, Ey, Ez


def current_correction(particles, Nx, Ny, Nz):
    Jx, Jy, Jz = jnp.zeros((Nx, Ny, Nz)), jnp.zeros((Nx, Ny, Nz)), jnp.zeros((Nx, Ny, Nz))
    # initialize the current arrays as 0
    for species in particles:
        q = species.get_charge()
        # get the charge of the species

        zeta1, zeta2, eta1, eta2, xi1, xi2 = species.get_subcell_position()
        # get the particle positions

        dx, dy, dz = species.get_resolution()

        deltax = zeta2 - zeta1
        deltay = eta2 - eta1
        deltaz = xi2 - xi1

        zetabar = 0.5*(zeta1 + zeta2)
        etabar = 0.5*(eta1 + eta2)
        xibar = 0.5*(xi1 + xi2)
        # compute the displacement variables from Buneman, Vilasenor 1991

        ix, iy, iz = species.get_index()

        if jnp.any(iy+1 > Ny) or jnp.any(ix+1 > Nx) or jnp.any(iz+1 > Nz):
            print("AHHHHHHHHHHHHHHHHHHHHHH")

        Jx = Jx.at[ix, iy+1, iz+1].add( q*(deltax*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first x correction for charge conservation along i+1/2, j, k
        Jx = Jx.at[ix, iy, iz+1].add( q*(deltax*(1-etabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second x correction for charge conservation along i+1/2, j, k+1
        Jx = Jx.at[ix, iy+1, iz].add( q*(deltax*etabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third x correction for charge conservation along i+1/2, j+1, k
        Jx = Jx.at[ix, iy, iz].add( q*(deltax*(1-etabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth x correction for charge conservation along i+1/2, j, k

        Jy = Jy.at[ix+1, iy, iz+1].add( q*(deltay*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first y correction for charge conservation along i, j+1/2, k
        Jy = Jy.at[ix+1, iy, iz].add( q*(deltay*(1-zetabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second y correction for charge conservation along i+1, j+1/2, k
        Jy = Jy.at[ix, iy, iz+1].add( q*(deltay*zetabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third y correction for charge conservation along i, j+1/2, k+1
        Jy = Jy.at[ix, iy, iz].add( q*(deltay*(1-zetabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth y correction for charge conservation along i, j+1/2, k

        Jz = Jz.at[ix+1, iy+1, iz].add( q*(deltaz*(1-zetabar)*etabar + deltax*deltay*deltaz/12))
        # compute the first z correction for charge conservation along i+1, j+1, k+1/2
        Jz = Jz.at[ix, iy+1, iz].add( q*(deltaz*(1-zetabar)*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the second z correction for charge conservation along i, j+1, k+1/2
        Jz = Jz.at[ix+1, iy, iz].add( q*(deltaz*zetabar*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the third z correction for charge conservation along i+1, j, k+1/2
        Jz = Jz.at[ix, iy, iz].add( q*(deltaz*(1-zetabar)*(1-etabar) + deltax*deltay*deltaz/12))
        # compute the fourth z correction for charge conservation along i, j, k+1/2

    return Jx, Jy, Jz
    # return the current corrections