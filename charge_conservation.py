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

@jit
def current_correction(Jx, Jy, Jz, particles, dx, dy, dz):
    q = particles.get_charge()
    # get the charge of the particles

    zeta1, zeta2, eta1, eta2, xi1, xi2 = particles.get_subcell_position()
    # get the particle positions

    dx, dy, dz = particles.get_resolution()

    deltax = zeta2 - zeta1
    deltay = eta2 - eta1
    deltaz = xi2 - xi1

    zetabar = 0.5*(zeta1 + zeta2)
    etabar = 0.5*(eta1 + eta2)
    xibar = 0.5*(xi1 + xi2)
    # compute the displacement variables from Buneman, Vilasenor 1991

    ix, iy, iz = particles.get_index()

    Jx = Jx.at[ix, iy+1, iz+1].set( q*(deltax*zetabar*xibar + deltax*deltay*deltaz/12))
    # compute the first x correction for charge conservation along i+1/2, j, k
    Jx = Jx.at[ix, iy, iz+1].set( q*(deltax*(1-etabar)*xibar - deltax*deltay*deltaz/12))
    # compute the second x correction for charge conservation along i+1/2, j, k+1
    Jx = Jx.at[ix, iy+1, iz].set( q*(deltax*etabar*(1-xibar) - deltax*deltay*deltaz/12))
    # compute the third x correction for charge conservation along i+1/2, j+1, k
    Jx = Jx.at[ix, iy, iz].set( q*(deltax*(1-etabar)*(1-xibar) + deltax*deltay*deltaz/12))
    # compute the fourth x correction for charge conservation along i+1/2, j, k

    Jy = Jy.at[ix+1, iy, iz+1].set( q*(deltay*zetabar*xibar + deltax*deltay*deltaz/12))
    # compute the first y correction for charge conservation along i, j+1/2, k
    Jy = Jy.at[ix+1, iy, iz].set( q*(deltay*(1-zetabar)*xibar - deltax*deltay*deltaz/12))
    # compute the second y correction for charge conservation along i+1, j+1/2, k
    Jy = Jy.at[ix, iy, iz+1].set( q*(deltay*zetabar*(1-xibar) - deltax*deltay*deltaz/12))
    # compute the third y correction for charge conservation along i, j+1/2, k+1
    Jy = Jy.at[ix, iy, iz].set( q*(deltay*(1-zetabar)*(1-xibar) + deltax*deltay*deltaz/12))
    # compute the fourth y correction for charge conservation along i, j+1/2, k

    Jz = Jz.at[ix+1, iy+1, iz].set( q*(deltaz*(1-zetabar)*etabar + deltax*deltay*deltaz/12))
    # compute the first z correction for charge conservation along i+1, j+1, k+1/2
    Jz = Jz.at[ix, iy+1, iz].set( q*(deltaz*(1-zetabar)*(1-etabar) - deltax*deltay*deltaz/12))
    # compute the second z correction for charge conservation along i, j+1, k+1/2
    Jz = Jz.at[ix+1, iy, iz].set( q*(deltaz*zetabar*(1-etabar) - deltax*deltay*deltaz/12))
    # compute the third z correction for charge conservation along i+1, j, k+1/2
    Jz = Jz.at[ix, iy, iz].set( q*(deltaz*(1-zetabar)*(1-etabar) + deltax*deltay*deltaz/12))
    # compute the fourth z correction for charge conservation along i, j, k+1/2

    return Jx, Jy, Jz
    # return the current corrections